#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require flomat)

;; ----------------------------------------------------

(require "activation.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define heritable<%>
  (interface () crossover))

;; ----------------------------------------------------

(define mutation-rate (make-parameter 0.03))

;; ----------------------------------------------------

(define (mutate x)
  (if (> (random) (mutation-rate))
      x
      (- (random) (random))))

;; ----------------------------------------------------

(define-pointwise-unary mutate)

;; ----------------------------------------------------

(define (recomb A B)
  (matrix (for/vector ([r (flomat->vectors A)]
                       [s (flomat->vectors B)])
            (if (< (random) 0.5) r s))))
