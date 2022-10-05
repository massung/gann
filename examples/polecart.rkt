#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require racket/gui)

;; gann code
(require "../main.rkt")

;; SEE: https://coneural.org/florian/papers/05_cart_pole.pdf

;; define a few global constants
(define G  9.8)     ; gravity (m/s^2)
(define Mc 1.0)     ; mass of the cart (kg)
(define Mp 1.0)     ; mass of the pole (kg)
(define L  2.0)     ; length of the pole (m)
(define F  10.0)    ; force applied (N)
(define S  1/33)    ; time step (s)

;; drawing constants
(define W 200)
(define H 300)
(define Y (* H 4/5))

;; define the state of the polecart
(struct state [x dx theta dt])

;; create a new state with a small jerk applied to it
(define (new-state)
  (let ([initial-angle (* (- (random) (random)) (degrees->radians 5))])
    (state 0 0 initial-angle 0)))

;; convert the state into an input vector
(define (state->X st)
  (match-let ([(state x dx theta dt) st])
    (column x dx theta dt)))

;; return a new state with the given force applied
(define (step-state st [f 0])
  (match-let ([(state x dx theta dt) st])
    (let* ([s (sin theta)]
           [c (cos theta)]

           ; total mass
           (M (+ Mc Mp))

           ; theta'' (equation 23)
           [ddt (/ (+ (* G s) (* c (/ (- f (* Mp L dt dt s)) M)))
                   (* L (- 4/3 (/ (* Mp c c) M))))]

           ; x'' (equation 24)
           [ddx (/ (- (* Mp L (- (* dt dt s) (* ddt c))) f) M)])

      ; return the new state
      (state (+ x (* dx S))
             (+ dx (* ddx S))
             (+ theta (* dt S))
             (+ dt (* ddt S))))))

;; perform an action, return reward, state, and terminal
(define (perform st action)
  (match-let ([(state x dx theta dt) st])
    (let ([angle (abs (radians->degrees theta))])
      (values
       ; reward for an ok angle
       (- 20 angle)

       ; create the new state
       (case action
         [(0)  (step-state st (- F))]         ; apply force to move left
         [(1)  (step-state st (+ F))]         ; apply force to move right
         [(2)  (step-state st (- (* F 2)))]   ; apply a large force to move left
         [(3)  (step-state st (+ (* F 2)))]   ; apply a large force to move right
         [else (step-state st)])              ; do nothing
       
       ; the state is terminal when the pole falls past 70 degrees or cart is off screen
       (or (> angle 70) (not (< -6 x 6)))))))

;; create the dqn model
(define-seq-model polecart-model
  [(dense-layer 9 .relu!)
   (dense-layer 5 .sigmoid!)]
  #:inputs 4)

;; create the canvas
(define polecart%
  (class canvas%
    (init-field
     [dqn (new dqn%
               [model polecart-model]
               [do-action perform]
               [initial-state new-state]
               [state->X state->X]
               [population-size 50]
               [batch-size #f])])

    ; drawing the state
    (super-new
     [paint-callback
      (λ (canvas dc)
        (match-let ([(state x dx theta dt) (send dqn get-state)])
          (let ([x (+ (/ W 2) (* x 20))])   ; move to the center and scale
            (send dc set-pen "black" 1 'solid)
            (send dc set-brush "black" 'solid)
            (send dc draw-line 0 Y W Y)
            (send dc draw-rectangle (- x 25) (- Y 25) 50 25)

            ; draw the pole
            (let ([pole-lx (* 50 L (sin theta))]
                  [pole-ly (* 50 L (cos theta))])
              (send dc set-pen "red" 2 'solid)
              (send dc draw-line x (- Y 25) (+ x pole-lx) (- Y 25 pole-ly))))))])

    ; message to train agents and redraw
    (define/public (step-sim)
      (send dqn train-agents #:watch? #t)
      (send this refresh))))

;; create the window frame
(define frame
  (new (class frame%
         (super-new)

         ; polecart canvas
         (define polecart
           (new polecart% [parent this]))

         ; game loop
         (define timer
           (new timer%
                [interval 10]
                [notify-callback (λ () (send polecart step-sim))]))

         ; stop timer on close
         (define/augment (on-close)
           (send timer stop)))
       
       ; initialization fields
       [label "DQN Polecart"]
       [x 0]
       [y 0]
       [width W]
       [height H]
       [style '(no-resize-border)]))

;; run it
(module+ main
  (send frame show #t))
