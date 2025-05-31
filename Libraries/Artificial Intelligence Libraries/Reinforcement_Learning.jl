/// AI Reinforcement Learning Library
/// Provides tools for building and training RL agents
pub module ReinforcementLearning {
    /// Reinforcement learning environment
    pub trait Environment {
        fn reset(&mut self) -> Observation;
        fn step(&mut self, action: Action) -> (Observation, Reward, bool);
        fn action_space(&self) -> Space;
        fn observation_space(&self) -> Space;
    }

    /// Reinforcement learning agent
    pub trait Agent {
        fn act(&mut self, obs: &Observation) -> Action;
        fn learn(
            &mut self,
            transition: &Transition
        ) -> Result<(), LearningError>;
    }

    /// Deep Q-Network agent
    pub struct DQNAgent {
        q_network: NeuralNetwork,
        target_network: NeuralNetwork,
        replay_buffer: ReplayBuffer,
        gamma: f32,
        epsilon: f32,
        epsilon_decay: f32
    }

    impl Agent for DQNAgent {
        fn act(&mut self, obs: &Observation) -> Action {
            if rand::random::<f32>() < self.epsilon {
                self.action_space().sample()
            } else {
                let q_values = self.q_network.forward(obs);
                argmax(&q_values)
            }
        }

        fn learn(&mut self, transition: &Transition) -> Result<(), LearningError> {
            self.replay_buffer.store(transition.clone());
            
            if self.replay_buffer.size() < BATCH_SIZE {
                return Ok(());
            }
            
            let batch = self.replay_buffer.sample(BATCH_SIZE);
            let mut loss = 0.0;
            
            for transition in batch {
                let target = if transition.done {
                    transition.reward
                } else {
                    let next_q = self.target_network.forward(&transition.next_obs);
                    transition.reward + self.gamma * next_q.max()
                };
                
                let mut q_values = self.q_network.forward(&transition.obs);
                q_values[transition.action] = target;
                
                loss += self.q_network.train_batch(&transition.obs, &q_values);
            }
            
            self.epsilon *= self.epsilon_decay;
            
            // Update target network
            if self.replay_buffer.size() % TARGET_UPDATE == 0 {
                self.update_target_network();
            }
            
            Ok(())
        }
    }

    /// Policy Gradient agent
    pub struct PolicyGradientAgent {
        policy_network: NeuralNetwork,
        optimizer: Optimizer,
        gamma: f32
    }

    impl Agent for PolicyGradientAgent {
        fn act(&mut self, obs: &Observation) -> Action {
            let probs = self.policy_network.forward(obs);
            sample_from_probs(&probs)
        }

        fn learn(&mut self, episode: &[Transition]) -> Result<(), LearningError> {
            let mut returns = compute_returns(episode, self.gamma);
            normalize(&mut returns);
            
            let mut policy_loss = 0.0;
            
            for (transition, &ret) in episode.iter().zip(returns.iter()) {
                let probs = self.policy_network.forward(&transition.obs);
                let log_prob = probs[transition.action].ln();
                policy_loss += -log_prob * ret;
            }
            
            self.optimizer.step(&mut self.policy_network, policy_loss);
            Ok(())
        }
    }

    /// Training loop
    pub fn train<E: Environment, A: Agent>(
        env: &mut E,
        agent: &mut A,
        episodes: usize
    ) -> Vec<f32> {
        let mut rewards = Vec::new();
        
        for _ in 0..episodes {
            let mut obs = env.reset();
            let mut total_reward = 0.0;
            let mut done = false;
            let mut transitions = Vec::new();
            
            while !done {
                let action = agent.act(&obs);
                let (next_obs, reward, done) = env.step(action);
                
                transitions.push(Transition {
                    obs: obs.clone(),
                    action,
                    reward,
                    next_obs: next_obs.clone(),
                    done
                });
                
                total_reward += reward;
                obs = next_obs;
            }
            
            for transition in &transitions {
                agent.learn(transition).unwrap();
            }
            
            rewards.push(total_reward);
        }
        
        rewards
    }
}