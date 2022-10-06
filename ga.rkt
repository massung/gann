#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require racket/generic)

;; ----------------------------------------------------

(require flomat)

;; ----------------------------------------------------

(require "activation.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

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

(define (crossover A B)
  (matrix (for/vector ([r (flomat->vectors A)]
                       [s (flomat->vectors B)])
            (if (< (random) 0.5) r s))))

;; ----------------------------------------------------

(define recomb<%>
  (interface () recomb))

;; ----------------------------------------------------

(define (next-gen! pop fitness [less-than? <])
  (let ([xs (vector-map (λ (x) (cons x (fitness x))) pop)])
    (vector-sort! xs less-than? #:key cdr)

    ; update the population, keep the best models
    (let ([n (quotient (vector-length xs) 10)])
      (for ([(x i) (in-indexed xs)])
        (if (< i n)
            (vector-set! pop i (car x))

            ; create new models
            (let ([a (car (vector-ref xs (random n)))]
                  [b (car (vector-ref xs (random n)))])
              (vector-set! pop i (send a recomb b))))))

    ; return the best fitness
    (cdr (vector-ref xs 0))))
