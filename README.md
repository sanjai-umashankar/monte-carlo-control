# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program for Monte Carlo control algorithm for given data

## PROBLEM STATEMENT
Develop a Python program that implements a Monte Carlo control algorithm to find the optimal policy for navigating the FrozenLake environment. The program should initialize the environment, define parameters (discount factor, learning rate, exploration rate), and implement decay schedules for efficient learning. It must generate trajectories based on an epsilon-greedy action selection strategy and update the action-value function using sampled episodes. Evaluate the learned policy by calculating the probability of reaching the goal state and the average undiscounted return. Finally, print the action-value function, state-value function, and optimal policy.

## MONTE CARLO CONTROL ALGORITHM
1. Initialize Q(s, a) arbitrarily for all state-action pairs
2. Initialize returns(s, a) to empty for all state-action pairs
3. Initialize policy π(s) to be arbitrary (e.g., ε-greedy)

4. For each episode:<BR>
   a. Generate an episode using policy π<BR>
   b. For each state-action pair (s, a) in the episode:<BR>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; i.   Calculate G (return) for that (s, a) pair<BR>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ii.  Append G to returns(s, a)<BR>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; iii. Calculate the average of returns(s, a)<BR>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; iv.  Update Q(s, a) using the average return<BR>
   c. Update policy π(s) based on Q(s, a)

## MONTE CARLO CONTROL FUNCTION
```
def mc_control(env, gamma=1.0, 
               init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):
    
    nS, nA = env.observation_space.n, env.action_space.n
    
    # Initialize Q-table, V-table, and policy (e.g., uniform random policy)
    Q = np.zeros((nS, nA))
    V = np.zeros(nS)
    pi = np.ones((nS, nA)) / nA
    
    Q_track = []
    pi_track = []
    
    # Get schedules for alpha and epsilon
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    # Pre-compute discount factors for steps
    discounts = np.logspace(0, max_steps, base=gamma, num=max_steps, endpoint=False)
    
    for e in range(n_episodes):
        alpha = alphas[e]
        epsilon = epsilons[e]
        
        # Generate trajectory using epsilon-greedy policy
        trajectory = generate_trajectory(select_action, Q, epsilon, env, max_steps)
        
        # Keep track of visited states/state-actions for first-visit vs every-visit MC
        visited = set()
        
        # Loop through trajectory backwards to calculate returns efficiently
        G = 0
        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[t]
            state = int(state)
            action = int(action)
            
            # Update return G
            G = gamma * G + reward
            
            if first_visit:
                if (state, action) not in visited:
                    visited.add((state, action))
                    # Update Q-value using incremental mean or alpha-based update
                    Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])
            else:
                Q[state, action] = Q[state, action] + alpha * (G - Q[state, action])
        
        # Update policy and state-value function V based on updated Q
        for s in range(nS):
            best_action = np.argmax(Q[s])
            # Epsilon-greedy policy update
            for a in range(nA):
                if a == best_action:
                    pi[s, a] = 1 - epsilon + (epsilon / nA)
                else:
                    pi[s, a] = epsilon / nA
            
            # State value is the expected value under the greedy policy (or max Q)
            V[s] = np.max(Q[s])
            
        Q_track.append(Q.copy())
        pi_track.append(pi.copy())
        
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
### Name: SANJAI  U
### Register Number: 212224240145

<img width="862" height="282" alt="image" src="https://github.com/user-attachments/assets/f4c4b832-6bdc-430f-98f6-2ea9b446a2f6" />


## RESULT:

Thus, the Python program has been successfully implemented.
