#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require flomat)

;; ----------------------------------------------------

(require "activation.rkt")
(require "dnn.rkt")
(require "layer.rkt")
(require "loss.rkt")
(require "model.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define explore%
  (class dnn%
    (init model)

    ; constructor fields
    (init-field [replay-size 50]
                [loss mean-squared])

    ; replay buffer and count
    (define replay '())
    (define replay-count 0)

    ; define a new model that attempts to predict state from action
    (let* ([m (model)]

           ; get the shape of the model
           [i (send m get-inputs)]
           [o (send m get-outputs)]

           ; hidden layer size
           [n (ceiling (* (+ i i o) 2/3))])

      ; the curiosity model models ((S,S') -> A), where A is hot-encoded
      (define-seq-model curiosity-model
        [(dense-layer n .relu!)
         (dense-layer o .step!)]
        #:inputs (+ i i))

      ; initialize with the right model
      (super-new [model curiosity-model]))

    ; access to superclass methods
    (inherit call get-model)

    ; add replay and return curiosity reward
    (define/public (get-curiosity X+Y K)
      (begin0 (loss K (call X+Y))
              
              ; add replay
              (set! replay (cons (list X+Y K) replay))
              (set! replay-count (add1 replay-count))
              
              ; train once the replay buffer is full
              (when (>= replay-count replay-size)
                (train))))

    ; update the curiosity model with replay data
    (define/override (train)
      (let-values ([(Xs Ys) (for/fold ([Xs '()]
                                       [Ys '()])
                                      ([r replay])
                              (values (cons (first r) Xs)
                                      (cons (second r) Ys)))])
        (super train (fit-training-data Xs Ys))
          
        ; reset replay buffer
        (set! replay '())
        (set! replay-count 0)))))

;; ----------------------------------------------------

(define (hot-encode i n)
  (apply column (for/list ([j n])
                  (if (= i j) 1.0 0.0))))

;; ----------------------------------------------------

(define (greedy Z)
  (for/fold ([k #f]
             [q #f] #:result k)
            ([(i z) (in-col Z 0)])
    (if (or (not q) (> z q))
        (values i z)
        (values k q))))

;; ----------------------------------------------------

(define (epsilon-greedy Z [epsilon 0.05])
  (if (< (random) epsilon)
      
      ; choose a random action
      (random (nrows Z))
      
      ; fallback to the "best" action
      (greedy Z)))

;; ----------------------------------------------------

(define (boltzmann Z [tau 0.3])
  (let ([n (random)])
    (for/fold ([k #f]
               [q 0.0] #:result k)
              ([(i z) (in-col (.softmax (./ Z tau)) 0)] #:break (< n q))
      (values i (+ q z)))))

;; ----------------------------------------------------

(module+ test
  (require rackunit)
  (require plot)

  ; hot encoding
  (check-equal? (hot-encode 2 5) (column 0 0 1 0 0))

  ; ensure action selection works
  (check-equal? (greedy (column 0 1 2 3)) 3)
  (check-equal? (greedy (column 3 2 1 0)) 0)

  ; a random set of action weights
  (define Z (column -0.1 0.2 0.6 0.3 0.1))

  ; generate histograms of exploration functions
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed 0)

    ; ensure boltzmann has the proper distribution
    (let* ([n 1000]
           [R (make-vector (size Z) 0)])
      (for ([_ n])
        (let ([i (boltzmann Z 0.5)])
          (vector-set! R i (add1 (vector-ref R i)))))
      
      ; plot how often each index was selected
      (let ([distribution (for/list ([x (flomat->vector Z)] [y R]) (list x y))])
        (plot (discrete-histogram distribution #:y-max n))))))
