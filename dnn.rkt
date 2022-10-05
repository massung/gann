#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require racket/generic)

;; ----------------------------------------------------

(require flomat)
(require plot)

;; ----------------------------------------------------

(require "activation.rkt")
(require "ga.rkt")
(require "layer.rkt")
(require "loss.rkt")
(require "model.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define dnn%
  (class object%
    (super-new)

    ; constructor fields
    (init-field model [population-size 100])

    ; build the initial population
    (field [models (build-vector population-size (λ (_) (model)))])

    ; return the best model
    (define/public (get-model)
      (vector-ref models 0))

    ; send inputs to the best model
    (define/public (call X)
      (send (get-model) call X))

    ; train a single generation, return the loss and the model
    (define/public (train fitness)
      (println 'here)
      (next-gen! models fitness <))

    ; train network models
    (define/public (train+ fitness #:generations [generations 100])
      (let ([loss (for/list ([x generations])
                    (list x (train fitness)))])
        (plot (lines loss #:y-min 0) #:x-label "Generation" #:y-label "Loss")))))

;; ----------------------------------------------------

(define (fit-training-data Xs Ys [loss mean-squared])
  (λ (model) 
    (for/sum ([X Xs] [Y Ys])
      (let ([Z (send model call X)])
        (loss Y Z)))))

;; ----------------------------------------------------

(define (training-data data)
  (for/fold ([Xs '()] [Ys '()])
            ([in/out data])
    (match in/out
      [(list in out)
       (values (cons (apply column in) Xs)
               (cons (apply column out) Ys))])))

;; ----------------------------------------------------

(module+ test  
  (define-seq-model xor-model
    [(dense-layer 5 .relu!)
     (dense-layer 1 .gaussian!)]
    #:inputs 2)

  ; create the xor neural network
  (define dnn (new dnn% [model xor-model]))

  ; construct training data set (X=input, Y=output)
  (define-values (Xs Ys)
    (training-data '([(0 0) (0)]
                     [(0 1) (1)]
                     [(1 0) (1)]
                     [(1 1) (0)])))

  ; train the network
  (time (send dnn train+ (fit-training-data Xs Ys)))
  
  ; run the test data to see the outputs
  (for ([X Xs])
    (let ([Z (send dnn call X)])
      (displayln (format "~a -> ~a" X Z)))))
