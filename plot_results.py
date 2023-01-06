# File that can be used to plot visualizations of Return and various other quantities

import matplotlib.pyplot as plt 
import seaborn as sns 

import os
import numpy as np 
import pickle 

def visualize_value(data_path,shape = (7,10),quantity='value'):
    # Helper function to visualize heatmaps of value function and delta function
    #for file in os.listdir(data_path):
    #    if quantity == 'value':
    #        if 'value' in file:
     #           q_values = np.load(data_path+file)
      #          values = np.max(q_values,axis=1).reshape(shape) # Extract value from Q values
       # else:
        #    if 'delta' in file:
         #       values = np.load(data_path+file, allow_pickle=True).reshape(shape)
    
    if quantity=='delta':
        values = np.load(data_path + 'delta.npy').reshape(shape)
    fig, ax = plt.subplots()
    im = ax.imshow(values,cmap=plt.get_cmap("viridis"), origin='lower')
    for i in range(shape[0]):
        for j in range(shape[1]):
            text = ax.text(j, i, round(values[i,j],1),
                       ha="center", va="center", color="w")

    plt.ylim(shape[0]-0.5, -0.5)
    if quantity=='value':
        title = 'Heatmap of Value function'
    elif quantity=='delta':
        title = 'Heatmap of Delta function'
    plt.title(title)
    plt.show()

def visualize_policy(data_path = 'data/', shape = (20,20)):
    # Helper function to visualize policy
    filename = data_path + 'policy_with_bias.npy'
    policy = np.load(filename)
    
    action2xy = {0:(-1,0), 1:(0,1), 2: (1,0), 3:(0,-1)}
    
    U = [action2xy[action][0] for action in policy] 
    V = [action2xy[action][1] for action in policy]
     
    # Arrow location
    X = np.arange(0, 20, 1)
    Y = np.arange(0, 20, 1)
    
    # Arrows X and Y components respectively
    # U, V = np.ones((len(X), len(Y))), np.zeros((len(X), len(Y)))
    U = np.asarray(U).reshape(shape)
    V = np.asarray(V).reshape(shape)
    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V)
    
    ax.quiverkey(q, X=0.3, Y=1.1, U=2,
                 label='Direction corresponding to optimal policy', labelpos='E')
    
    plt.ylim(shape[0]-0.5, -0.5)
    plt.show()
    
def plot_returns(result_path,algo = 'qlearning'):
    # Helper function to plot return values after taking a running average
    return_results = {}
    window = 100 # Window size for running average

    for file in os.listdir(result_path):

        if algo in file:
            with open(result_path + file, 'rb') as f:
                episode_returns, method_name = pickle.load(f)
            return_results.update({method_name: episode_returns}) 
    
    #return_results.update({'Target Optimal':np.ones(500)*53})
    for key, result in return_results.items():

        running_average = np.convolve(result, np.ones(window)*1/window, mode="valid")
        x = np.arange(window, running_average.shape[0]+window)
        sns.lineplot(x, running_average, label=key)

    plt.title('Transfer results across different dynamics')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Episodic Returns')
    plt.show()

if __name__ == '__main__':

    plot_type = 'return' # 'return' : for plotting returns, 'value': To visualize value functions , 'delta': To visualize delta function, 'policy' : To visualize policy 

    if plot_type == 'return':
        result_path = 'results/delta_inventory/'  # Path to folder containing results
        plot_returns(result_path = result_path,algo = 'qlearning')
    elif plot_type == 'value': 
        data_path = 'results/delta_similarity/' # Path to folder containing stored Q values and delta
        visualize_value(data_path=data_path,shape=(20,20),quantity = 'value') 
    elif plot_type == 'delta': 
        data_path = 'results/delta_inventory/' # Path to folder containing stored Q values and delta
        visualize_value(data_path=data_path,shape=(20,20),quantity = 'delta') 
    else:
        data_path = 'results/policy/' # Path to folder containing  stored policy
        visualize_policy(data_path = data_path)