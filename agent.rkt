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
  (class* object% (recomb<%>)
    (super-new)

    ; constructor fields
    (init-field model
                initial-state
                state->X
                perform-action)

    ; maintain current state, total reward, and terminal state flag
    (define state (initial-state))
    (define reward 0)
    (define terminal? #f)
    #;(field [state (initial-state)]
           [reward 0]
           [terminal? #f])

    (define/public (agent-terminal?) terminal?)
    (define/public (agent-reward) reward)
    (define/public (agent-state) state)

    (define/public (show)
      (displayln (format "xx ~a" reward)))
    
    ; reset 
    (define/public (reset-state)
      (set! state (initial-state))
      (set! reward 0)
      (set! terminal? #f))

    ; run input vector
    (define/public (step [train? #f])
      (or terminal?

          ; choose action, perform, and update state
          (let* ([X (state->X state)]
                 [Z (send model call X)]
                 [k (epsilon-greedy Z 0.05)])
            (let-values ([(r new-state new-terminal?)
                          (perform-action state k)])
              (begin0 new-terminal?
                      
                      ; update fields
                      (set! state new-state)
                      (set! terminal? new-terminal?)
                      (set! reward (+ reward (* r #;(expt gamma batch)))))))))
    
    ; recombine with another agent
    (define/public (recomb other)
      (new this%
           [model (send model recomb (get-field model other))]
           [initial-state initial-state]
           [state->X state->X]
           [perform-action perform-action]))))
