import gym
import numpy as np
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)









class OneStepActorCritic:

    
    
    def __init__(self,env,num_actions,num_states,M,alpha_theta,alpha_weights,gamma,low,high,acrobot_true):
    
    
        ### Initialize number of actions ###
        self.num_actions = num_actions
    
        ### Initialize fourier basis order ###
        self.Fourier_order = M
        
        ### Initialize policy_parameters |A|*|phi(s)| ###
        #print("num_states = ",num_states)
        #print("num_actions = ",num_actions)
        self.policy_parameters = np.zeros((num_actions,num_states*M+1))
        
        ### Initialze weights for the value function ###
        
        self.weights = np.zeros(num_states*M+1)
        
        ### Initialize learning rates ###
        self.alpha_theta = alpha_theta
        self.alpha_weights = alpha_weights
        
        ### Initialize discount factor gamma ###
        self.gamma = gamma 
        
        ### Initialize low and high of the state of the env
        self.low = low 
        self.high = high
        
        ### Set whether env is acrobot or mountain car 
        self.acrobot_true = acrobot_true
        
    
    def fourier_basis(self,state):  
        
        ### total terms for each position in state ###
        terms = np.arange(1, self.Fourier_order + 1)

        ### Normalize state ###
        normalized_state = (state - self.low)/(self.high - self.low) 
 
        '''
        ### Create state features array ###
        state_features = np.concatenate([
        np.ones(1),  # The first term is always 1
        np.cos(terms * np.pi * normalized_state[0]),
        np.cos(terms * np.pi * normalized_state[1]),
        np.cos(terms * np.pi * normalized_state[2]),
        np.cos(terms * np.pi * normalized_state[3]),
        np.cos(terms * np.pi * normalized_state[4]),
        np.cos(terms * np.pi * normalized_state[5])
        ])
        
        '''
        
        state_features = []
        state_features.append(1)
        
        for s in normalized_state:
            state_features.extend(np.cos(terms * np.pi * s))
        
        #print(state_features)
        
        return np.array(state_features)
       

    def select_action(self,state):
    
        ### Calculate state features for the given state ###
        current_state_features = self.fourier_basis(state)
        
        ### Get the parameterized policy ###
        policy = np.exp(np.dot(self.policy_parameters,current_state_features))
        #print(policy.shape)
        
        ### Calculate probability of each action ###
        action_probabilities = policy/np.sum(policy)
        #print("action_probabilities = ",action_probabilities)
        
        ### Choose an action randomly ###
        action = np.random.choice(len(policy), p=action_probabilities)
        
        
        return action,current_state_features,action_probabilities
    
    def gradient_ln_policy(self,action_probs,Action,state_features):
     
        policy_grad_vector = []
        for a in range(self.num_actions):
            if a == Action:
                policy_grad_vector.append((1-action_probs[a])*state_features)
            else:
                policy_grad_vector.append(-action_probs[a]*state_features)
           
        return policy_grad_vector
        
    def actor_critic_algorithm(self,env,episodes):
        
        rewards_plot = []
       
        for episode in range(episodes):
            
            #### Initialze state of the environment ####
            state = env.reset(seed=453)[0] 
           
            ### Initialize I and reward ###
            I = 1
            episode_total_reward = 0
            step = 1
            
            while True:
            
                #### Select an action from policy ####
                action,current_state_features,current_state_action_probabilities = self.select_action(state)
                
                #### Apply action to the env and get the next state and reward ###
                next_state, reward, done, _ , _ = env.step(action)
                #print("next_state = ",next_state)
                
                
                
                #### Caluculate TD error delta ####
                # Calculate next_state features 
                next_state_features = self.fourier_basis(next_state)
                # Calculate current and next_state values  
                v_current_state = np.dot(self.weights,current_state_features)
                v_next_state = np.dot(self.weights,next_state_features)
                delta = reward + self.gamma*v_next_state - v_current_state
                
                
                #### Train the model --> adjust policy_parameters and weights ####
                # Set weights 
                self.weights += self.alpha_weights*delta*current_state_features
                
                # Set policy_parameters theta 
                policy_grad_vector = np.array(self.gradient_ln_policy(current_state_action_probabilities,action,current_state_features))
                #print(np.array(policy_grad_vector))

                self.policy_parameters += self.alpha_theta*I*delta*policy_grad_vector
                
                
                # Modify I 
                I = self.gamma*I
                
                # Set next state as curent state and add reward to total_reward
                state = next_state
                episode_total_reward += reward 
                
                #### episode terminates if done = TRue, i.e., acrobot has crossed the line
                if done:
                    print(f"Episode Number: {episode + 1}, Total Reward Earned: {episode_total_reward}, Note: Episode Terminated")
                    rewards_plot.append(episode_total_reward)
                    break
                
                #### else episode terminates if number of steps > 500 fro acrobot and > 200 for mountain car
                if self.acrobot_true:
                    if step >= 500:
                        print(f"Episode Number: {episode + 1}, Total Reward Earned: {episode_total_reward}, Note: Episode Truncated")
                        rewards_plot.append(episode_total_reward)
                        break
                else:
                    if step >= 200:
                        print(f"Episode Number: {episode + 1}, Total Reward Earned: {episode_total_reward}, Note: Episode Truncated")
                        rewards_plot.append(episode_total_reward)
                        break
                    
                
                step +=1
        
        return rewards_plot

                              
if __name__ == '__main__':



    ##############################################################
    ############# 1 STep Actor-Critic for Acrobot ################
    ##############################################################

    print("\n##################################################################")
    print("#######    Implementing One-Step Actor-Critic ON ACROBOT    ######")
    print("##################################################################")

    #### Environmnet definition - Acrobot ####
    env = gym.make("Acrobot-v1")
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    #env.seed(2000)
      
    ### Set order of Fourier basis ###
    M_acrobot = [5,3,6,7,10]   
    
    ### Run for these many episodes ###
    episodes = 1000
    
    ### Set learning rates for actor (policy) and critic(value Function) ###
    alpha_theta = [0.01, 0.001, 0.01, 0.0001, 0.02]
    alpha_weights = [0.01, 0.001, 0.025, 0.00025, 0.025]
    #alpha_theta = 0.001
    #alpha_weights = 0.001
    
    ### Set the discount rate gamma ###
    gamma = [0.99, 0.99, 0.99, 0.98, 0.99]
    #gamma = 1
    
    ### Set states low and high for Acrobot
    low = np.array([-1, -1, -1, -1, -12.566371, -28.274334])
    high = np.array([1, 1, 1, 1, 12.566371, 28.274334])
       
    ### Call Actor Critic ###
    rewards_over_hyperparameters = []
    for i in range(5):
        print("\nFor Discount Rate (γ) = ",gamma[i]," Actor Learning Rate = ",alpha_theta[i]," Critic Learning rate = ",alpha_weights[i]," Fourier Order(M) = ",M_acrobot[i])
        ActorCritic = OneStepActorCritic(env,num_actions,num_states,M_acrobot[i],alpha_theta[i],alpha_weights[i],gamma[i],low,high,True)
        r = ActorCritic.actor_critic_algorithm(env,episodes)
        rewards_over_hyperparameters.append(r)
        
    ### Close the environment ###
    env.close()
    
    
    
    ##################################################################
    ############# 1 STep Actor-Critic for MountainCar ################
    ##################################################################

    
    print("\n##################################################################")
    print("######  Implementing One-Step Actor-Critic On MOuntain Car  ######")
    print("##################################################################")

    #### Environmnet definition - MountainCar ####
    env_mc = gym.make("MountainCar-v0")
    num_states_mc = env_mc.observation_space.shape[0]
    num_actions_mc = env_mc.action_space.n
    #env.seed(2000)
      
    ### Set order of Fourier basis ###
    M_MountainCar = [4,3,6,10,9]   
    
    ### Run for this many episodes ###
    episodes_mc = 1000
    
    ### Set learning rates for actor (policy) and critic(value Function) ###
    #alpha_theta = [0.01, 0.001, 0.005, 0.5, 1]
    #alpha_weights = [0.01, 0.001, 0.005, 0.5, 1]
    alpha_theta_mc = [0.01, 0.001, 0.01, 0.1, 0.02]
    alpha_weights_mc = [0.0001, 0.001, 0.025, 0.1, 0.025]
    
    ### Set discount_rate gamma ###
    #gamma = [0.99, 1, 0.8, 1, 0.5]
    gamma_mc = [0.99, 0.99, 0.99, 0.98, 0.99]
    
    ### Set states low and high for MountainCar
    low_mc = np.array([-1.2, -0.07])
    high_mc = np.array([0.6,0.07])
       
    ### Call Actor Critic ###
    rewards_over_hyperparameters_mc = []
    for i in range(5):
        print("\nFor Discount Rate (γ) = ",gamma_mc[i]," Actor Learning Rate = ",alpha_theta_mc[i]," Critic Learning rate = ",alpha_weights_mc[i]," Fourier Order(M) = ",M_MountainCar[i])
        ActorCritic_mc = OneStepActorCritic(env_mc,num_actions_mc,num_states_mc,M_MountainCar[i],alpha_theta_mc[i],alpha_weights_mc[i],gamma_mc[i],low_mc,high_mc,False)
        r_mc = ActorCritic_mc.actor_critic_algorithm(env_mc,episodes_mc)
        rewards_over_hyperparameters_mc.append(r_mc)
        
    ### Close the environment ###
    env_mc.close()
    
    
    #################################################
    ################# Plot Graphs ###################
    #################################################
    
    
    ############## ACROBOT ##################
  
    line_style = '--'
    custom_x_ticks = np.arange(0, episodes+1, 100)
    #ax = plt.gca()
   # ax.axhline(y=-500, color='black', linestyle='--', label='Threshold')
    #plt.figure(1,figsize=(10, 6))
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True) 
    fig.subplots_adjust(hspace=0.5)
    axs = axs.flatten()
    for j in range(5):
        std_dev_acr = np.std(rewards_over_hyperparameters[j],axis=0)
        axs[j].axhline(y=-500, color='red', linestyle='--', label='Threshold at -500')
        axs[j].plot(range(1, episodes + 1), rewards_over_hyperparameters[j], linestyle = line_style, label= f'γ = {gamma[j]} α^θ = {alpha_theta[j]} α^w = {alpha_weights[j]} M = {M_acrobot[j]}')
        axs[j].fill_between(range(1, episodes + 1), rewards_over_hyperparameters[j] - std_dev_acr, rewards_over_hyperparameters[j] + std_dev_acr, alpha=0.4, color='lightblue', label="Standard Deviation")
        axs[j].set_xlabel("Number of Episodes")
        axs[j].set_ylabel("Reward")
        axs[j].set_xticks(custom_x_ticks)
        axs[j].set_title(f"Learning Curve for ACROBOT : Hyperparameter {j+1}")
        axs[j].legend(loc='upper left',bbox_to_anchor=(1, 1))
        axs[j].grid(True)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    ############## MOUNTAIN_CAR ##################
  
    line_style_mc = '--'
    custom_x_ticks_mc = np.arange(0, episodes_mc+1, 100)
    fig_mc, axs_mc = plt.subplots(5, 1, figsize=(12, 12), sharex=True) 
    fig_mc.subplots_adjust(hspace=0.5)
    axs_mc = axs_mc.flatten()
    for j in range(5):
        std_dev_mc = np.std(rewards_over_hyperparameters_mc[j],axis=0)
        axs_mc[j].axhline(y=-200, color='red', linestyle='--', label='Threshold at -200')
        axs_mc[j].plot(range(1, episodes_mc + 1), rewards_over_hyperparameters_mc[j], linestyle = line_style_mc, label= f'γ = {gamma_mc[j]} α^θ = {alpha_theta_mc[j]} α^w = {alpha_weights_mc[j]} M = {M_MountainCar[j]}')
        axs_mc[j].fill_between(range(1, episodes_mc + 1), rewards_over_hyperparameters_mc[j] - std_dev_mc, rewards_over_hyperparameters_mc[j] + std_dev_mc, alpha=0.4, color='lightblue', label="Standard Deviation")
        axs_mc[j].set_xlabel("Number of Episodes")
        axs_mc[j].set_ylabel("Reward")
        axs_mc[j].set_xticks(custom_x_ticks_mc)
        axs_mc[j].set_title(f"Learning Curve for MOUNTAIN_CAR: Hyperparameter {j+1}")
        axs_mc[j].legend(loc='upper left',bbox_to_anchor=(1, 1))
        axs_mc[j].grid(True)
    
    # Adjust layout for better spacing
    plt.tight_layout()

    
    
    ##### Plotting the best Hyperparameter Leaning Plot ##########
    
    ##### For ACROBOT #####
    custom_x_ticks_acrobot_best = np.arange(0, episodes+1, 100)
    plt.figure(3,figsize=(10, 6))
    std_dev_acr_best = np.std(rewards_over_hyperparameters[1],axis=0)
    mean_value_acrobot_best = np.mean(rewards_over_hyperparameters[1])
    plt.axhline(y=mean_value_acrobot_best, color='purple', linestyle='-', label=f'Mean: {mean_value_acrobot_best:.2f}')
    plt.axhline(y=-500, color='red', linestyle='--', label='Threshold at -500')
    plt.plot(range(1, episodes + 1), rewards_over_hyperparameters[1], linestyle = '-', label= f'γ = {gamma[1]} α^θ = {alpha_theta[1]} α^w = {alpha_weights[1]} M = {M_acrobot[1]}')
    plt.fill_between(range(1, episodes + 1), rewards_over_hyperparameters[1] - std_dev_acr_best, rewards_over_hyperparameters[1] + std_dev_acr_best, alpha=0.4, color='lightblue', label="Standard Deviation")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Reward")
    plt.xticks(custom_x_ticks_acrobot_best)
    plt.title(f"Learning Curve for ACROBOT for Best Hyperparameter with Mean and Standard Deviation")
    plt.legend(loc='upper left')
    plt.grid(True)
    
    ##### For MountainCar #####
    custom_x_ticks_mc_best = np.arange(0, episodes_mc+1, 100)
    plt.figure(4,figsize=(10, 6))
    std_dev_mc_best = np.std(rewards_over_hyperparameters_mc[4],axis=0)
    mean_value_mc_best = np.mean(rewards_over_hyperparameters_mc[4])
    plt.axhline(y=mean_value_mc_best, color='purple', linestyle='-', label=f'Mean: {mean_value_mc_best:.2f}')
    plt.axhline(y=-200, color='red', linestyle='--', label='Threshold at -200')
    plt.plot(range(1, episodes_mc + 1), rewards_over_hyperparameters_mc[4], linestyle = '-', label= f'γ = {gamma_mc[4]} α^θ = {alpha_theta_mc[4]} α^w = {alpha_weights_mc[4]} M = {M_MountainCar[4]}')
    plt.fill_between(range(1, episodes_mc + 1), rewards_over_hyperparameters_mc[4] - std_dev_mc_best, rewards_over_hyperparameters_mc[4] + std_dev_mc_best, alpha=0.4, color='lightblue', label="Standard Deviation")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Reward")
    plt.xticks(custom_x_ticks_mc_best)
    plt.title(f"Learning Curve for MOUNTAIN CAR for Best Hyperparameter with Mean and Standard Deviation")
    plt.legend(loc='upper left')
    plt.grid(True)
    


    
    plt.show()
    