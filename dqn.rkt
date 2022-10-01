#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require flomat)
(require plot)

;; ----------------------------------------------------

(require "activation.rkt")
(require "agent.rkt")
(require "dnn.rkt")
(require "ga.rkt")
(require "layer.rkt")
(require "loss.rkt")
(require "model.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(struct agent [rewards state terminal?] #:mutable)

;; ----------------------------------------------------

(define dqn%
  (class dnn%
    (init model)
    
    ; constructor fields
    (init-field initial-state
                state->X
                perform

                ; optional test for valid actions
                #;[valid-action? (λ (st k) #t)]

                ; learning fields
                [batch-size 100]
                [gamma 0.9])

    ; batch count for this generation
    (field [batch 0])
    
    ; initialize the network using agent models
    (super-new [model (λ () (new agent%
                                 [model (model)]
                                 [initial-state initial-state]))])

    ; rename the models to agents
    (inherit-field [agents models])

    ; rename get-model to get-agent 
    (inherit [get-agent get-model])

    ; lookup the state for an agent
    (define/public (get-state [i 0])
      (get-field state (get-agent i)))

    ; reset the state for an agent
    (define/public (reset-state [i 0])
      (send (get-agent i) reset-state))

    ; perform an action with the best agent
    (define/override (predict [agent (get-agent)] #:explore [explore greedy] #:train? [train? #f])
      (send agent predict perform state->X #:explore explore #:train? train?))
    
    ; perform actions for all agents in the population
    (define/public (train-agents #:explore [explore greedy] #:watch? [watch? #f])
      (set! batch (add1 batch))

      ; advance all agents, is it the end of a generation?
      (and (or (for/fold ([all-terminal? #t])
                         ([agent agents])
                 (let ([terminal? (predict agent #:explore explore #:train? #t)])
                   (and all-terminal? terminal?)))

               ; batch full?
               (and batch-size (>= batch batch-size))

               ; watch one particular agent
               (and watch? (get-field terminal? (get-agent))))

           ; advance to the next generation
           (next-gen)))

    ; reset batch count and population a new generation of agents
    (define/override (next-gen)
      (begin0 (super next-gen (agent-fitness gamma) #:less-than? >)

              ; reset the batch count
              (set! batch 0)

              ; reset the state of the elite agent 
              (send (get-agent) reset-state)))

    ; train agents for multiple generations
    (define/override (train #:explore [explore greedy] #:generations [n 100])
      (let ([xy (for/list ([x n])
                  (do ([y (train-agents #:explore explore)
                          (train-agents #:explore explore)])
                    [y (list x y)]))])
        (plot (lines xy #:y-min 0) #:x-label "Generation" #:y-label "Reward")))))
