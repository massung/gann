#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require "../main.rkt")

;; create a simple xor model
(define-seq-model <xor-model>
  [(<dense-layer> 5 .relu!)
   (<dense-layer> 1 .step!)]
  #:inputs 2)

;; create the xor neural network
(define dnn (new dnn% [model <xor-model>]))

;; construct training data set (X=input, Y=output)
(define-values (Xs Ys)
  (training-data '([(0 0) (0)]
                   [(0 1) (1)]
                   [(1 0) (1)]
                   [(1 1) (0)])))

;; train the network
(let ([fitness (fit-training-data Xs Ys)])
  (send dnn train+ fitness))

;; run the test data to see the outputs
(for ([X Xs])
  (let ([Z (call (send dnn get-model) X)])
    (displayln (format "~a -> ~a" X Z))))
