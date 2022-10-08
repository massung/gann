#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require racket/gui)

;; ----------------------------------------------------

(require "../main.rkt")

;; ----------------------------------------------------

(define width 200)
(define height 300)
(define ball-speed 150)
(define paddle-speed 300)
(define radius 5)
(define paddle-size 30)
(define dt 33/1000)

;; ----------------------------------------------------

(define middle (/ width 2.0))
(define y (* height 4/5))

;; ----------------------------------------------------

(define (random-x [margin 0])
  (+ (* (random) (- width (* margin 2))) margin))

;; ----------------------------------------------------

(define (clamp-x x [margin 0])
  (max (min x (- width margin)) margin))

;; ----------------------------------------------------

(define (random-angle)
  (+ (/ pi 4) (* (/ pi 2) (random))))

;; ----------------------------------------------------

(struct state [x      ; paddle position
               bx     ; ball x
               by     ; ball y
               angle  ; ball direction
               speed  ; ball speed
               score  ; consecutive ceiling hits
               ]
  #:transparent)

;; ----------------------------------------------------
  
(define (new-state)
  (state middle (random-x radius) radius (- (random-angle)) ball-speed 0))

;; ----------------------------------------------------

(define (state->X st)
  (let ([dx (- (state-bx st) (state-x st))])
    (column (/ dx width 0.5)                 ; distance of ball to paddle [-1,1]
            (/ (state-angle st) pi))))       ; ball direction of travel [-1,1]

;; ----------------------------------------------------

(define (bounce angle fx fy)
  (atan (* fy (sin angle))
        (* fx (cos angle))))

;; ----------------------------------------------------

(define (perform st action)
  (let* ([speed (state-speed st)]

         ; velocity vector
         [vx (* (cos (state-angle st)) speed)]
         [vy (* (sin (state-angle st)) speed)]
         
         ; paddle position delta
         [dx (case action
               [(0) (- (* dt paddle-speed))] ; left
               [(1) (+ (* dt paddle-speed))] ; right
               [else 0])]

         ; new paddle pos clamped to edges
         [px (max (min (+ (state-x st) dx) width) 0)]

         ; new ball position
         [nx (+ (state-bx st) (* vx dt))]
         [ny (- (state-by st) (* vy dt))]

         ; collision with ceiling or paddle
         [hit-top? (< ny radius)]
         [hit-paddle? (and (<= (- y radius) ny (+ y radius))
                           (<= (- px paddle-size) nx (+ px paddle-size)))]

         ; off board?
         [loss? (> ny height)])

    ; return the reward and new state
    (values (- paddle-size (abs (- nx px))) #;(if hit-paddle? 1 0)

            ; new state
            (state px

                   ; new ball pos x
                   (clamp-x nx radius)

                   ; new ball pos y
                   (cond
                     [hit-paddle? (- y radius)]
                     [hit-top? radius]
                     [else ny])

                   ; new angle
                   (cond
                     [hit-paddle? (random-angle)]
                     
                     ; bounce off left/right walls
                     [(not (< radius nx (- width radius)))
                      (bounce (state-angle st) -1 1)]
                     
                     ; bounce off top
                     [hit-top? (bounce (state-angle st)  1 -1)]
                     
                     ; move in same direction
                     [else (state-angle st)])
                   
                   ; increase speed over time with a speed limit
                   (min (+ speed (if hit-top? 1 0)) 300)
                   
                   ;; update score
                   (+ (state-score st) (if hit-top? 1 0)))

            ;; terminal state on ball loss
            loss?)))

;; ----------------------------------------------------

(define-seq-model pong-model
  [(dense-layer 5 .relu!)
   (dense-layer 5 .relu!)
   (dense-layer 3 .relu!)]
  #:inputs 2)

;; ----------------------------------------------------

(define dqn (new dqn%
                 [model pong-model]
                 [population-size 50]
                 [initial-state new-state]
                 [state->X state->X]
                 [perform-action perform]
                 [batch-size #f]))

;; ----------------------------------------------------

(define frame
  (new (class frame%
         (super-new)

         ; playfield canvas
         (define canvas (new canvas%
                             [parent this]
                             [paint-callback (λ (canvas dc)
                                               (let ([st (send dqn get-state)])
                                                 (send dc set-brush "black" 'solid)
                                                 (send dc draw-text (format "SCORE: ~a" (state-score st)) 5 5)
                                                 (send dc set-brush "black" 'solid)
                                                 (send dc draw-rectangle (- (state-x st) paddle-size) y (* paddle-size 2) 5)
                                                 (send dc set-brush "red" 'solid)
                                                 (send dc draw-ellipse (- (state-bx st) radius) (- (state-by st) radius) (* radius 2) (* radius 2))))]))

         ; game loop timer
         (define timer (new timer%
                            [interval 5]
                            [notify-callback (λ ()
                                               (send dqn train #:watch? #t)
                                               (send canvas refresh))]))

         ; stop learning/playing
         (define/augment (on-close)
           (send timer stop)))

       ; initialization fields
       [label "DQN Pong"]
       [width width]
       [height height]
       [style '(no-resize-border)]))

;; ----------------------------------------------------
  
(module+ main
  (send frame show #t))
