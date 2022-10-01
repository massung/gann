#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require flomat)

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define (mean-loss .proc Y Z)
  (/ (for/sum ([i (size Y)])
       (.proc (- (ref Y i 0) (ref Z i 0))))
     (size Y)))

;; ----------------------------------------------------

(define mean-abs (curry mean-loss abs))
(define mean-squared (curry mean-loss (λ (x) (* x x))))

;; ----------------------------------------------------

(define (huber [threshold 1.0])
  (let ([half (* threshold 0.5)])
    (curry mean-loss (λ (x)
                       (let ([n (abs x)])
                         (if (> n threshold)
                             (* threshold (- n half))
                             (* n n)))))))

