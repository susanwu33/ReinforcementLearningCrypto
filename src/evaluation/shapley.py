import numpy as np
import torch
import pandas as pd

def calculate_shap_for_agent(policy_net, env, device, feature_names, n_background=1000, n_test=100):
    """
    Computes SHAP values for a PyTorch policy network.
    Returns the top 3 feature names by mean absolute SHAP value.
    """
    try:
        import shap
    except ImportError:
        print("Please install shap: pip install shap")
        return []

    # Gather background samples from random interactions
    obs, _ = env.reset()
    background = []
    for _ in range(n_background):
        a = env.action_space.sample()
        obs, _, done, _, _ = env.step(a)
        background.append(obs)
        if done:
            obs, _ = env.reset()
            
    background = torch.tensor(np.array(background), dtype=torch.float32).to(device)

    # Gather test samples
    test_obs = []
    obs, _ = env.reset()
    for _ in range(n_test):
        # We can simulate the policy or just randomly sample to get a diverse set of states
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if hasattr(policy_net, "net") and isinstance(policy_net.net[-1], torch.nn.Softmax):
                # REINFORCE policy
                probs = policy_net(x)
                a = torch.argmax(probs, dim=1).item()
            else:
                # DQN Q-net
                a = torch.argmax(policy_net(x), dim=1).item()
                
        obs, _, done, _, _ = env.step(a)
        test_obs.append(obs)
        if done:
            obs, _ = env.reset()

    test_obs = torch.tensor(np.array(test_obs), dtype=torch.float32).to(device)

    try:
        # Use DeepExplainer
        explainer = shap.DeepExplainer(policy_net, background)
        shap_values = explainer.shap_values(test_obs)
        
        if isinstance(shap_values, list):
            avg_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            avg_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
            
    except Exception as e:
        print(f"SHAP execution failed due to environment issues ({e}). Falling back to Marginal Perturbation Analysis.")
        
        with torch.no_grad():
            base_outputs = policy_net(test_obs)
            if base_outputs.dim() == 1:
                base_outputs = base_outputs.unsqueeze(1)
                
            avg_abs_shap = np.zeros(test_obs.shape[1])
            for i in range(test_obs.shape[1]):
                perturbed_obs = test_obs.clone()
                # Nullify the feature by setting it to background mean
                perturbed_obs[:, i] = background[:, i].mean()
                
                perturbed_outputs = policy_net(perturbed_obs)
                if perturbed_outputs.dim() == 1:
                    perturbed_outputs = perturbed_outputs.unsqueeze(1)
                
                # Use L1 diff in output distribution/Q-values as feature importance proxy
                importance = torch.mean(torch.abs(base_outputs - perturbed_outputs)).item()
                avg_abs_shap[i] = importance

    # Construct dataframe mapping features to importance
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": avg_abs_shap
    }).sort_values(by="importance", ascending=False)
    
    print("\nTop 5 features by SHAP importance:")
    print(imp_df.head(5))

    top_3 = imp_df.head(3)["feature"].tolist()
    return top_3, imp_df
