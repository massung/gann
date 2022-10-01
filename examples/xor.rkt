#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require "../main.rkt")

;; create a simple xor model
(define-model xor-model seq-model%
  [(dense-layer 5 .relu!)
   (dense-layer 1 .gaussian!)]
  #:inputs 2)

;; create the xor neural network
(define dnn (new dnn% [model xor-model]))

;; construct training data set (X=input, Y=output)
(define Xs '([0 0] [0 1] [1 0] [1 1]))
(define Ys '([0]   [1]   [1]   [0]))

;; train the network
(time (send dnn train Xs Ys))

;; run the test data to see the outputs
(for ([X Xs])
  (let ([Z (send dnn predict X)])
    (displayln (format "~a -> ~a" X Z))))
