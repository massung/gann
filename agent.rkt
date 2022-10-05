#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require flomat)

;; ----------------------------------------------------

(require "activation.rkt")
(require "explore.rkt")
(require "ga.rkt")
(require "model.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define agent%
  (class model%
    (super-new)

    ; constructor fields
    (init-field initial-state)

    ; maintain current state, total reward, and terminal state flag
    (field [state (initial-state)]
           [reward 0]
           [terminal? #f])

    ; create a new agent with a new model
    (define/override (recomb agent)
      (let ([new-model (super model crossover (get-field model agent))])
        (new this%
             [model new-model]
             [initial-state initial-state])))))
