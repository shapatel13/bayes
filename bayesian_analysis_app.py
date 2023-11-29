import streamlit as st
import numpy as np
import pymc as pm
import arviz as az

# Function to set Beta distribution parameters based on user selection
def set_prior_params(prior_type):
    if prior_type == 'Neutral':
        return 1, 1  # Beta(1, 1) for Neutral prior
    elif prior_type == 'Optimistic':
        return 2, 1  # Beta(2, 1) for Optimistic prior (leaning towards benefit)
    elif prior_type == 'Pessimistic':
        return 1, 2  # Beta(1, 2) for Pessimistic prior (leaning towards harm)

# Function for Bayesian analysis of binary data
def bayesian_analysis_binary(n_intervention, events_intervention, n_control, events_control, prior_type):
    alpha, beta = set_prior_params(prior_type)

    with pm.Model() as model_binary:
        # Beta priors for intervention and control
        prior_intervention = pm.Beta('prior_intervention', alpha=alpha, beta=beta)
        prior_control = pm.Beta('prior_control', alpha=alpha, beta=beta)

        # Binomial Likelihoods
        likelihood_intervention = pm.Binomial('likelihood_intervention', n=n_intervention, p=prior_intervention, observed=events_intervention)
        likelihood_control = pm.Binomial('likelihood_control', n=n_control, p=prior_control, observed=events_control)

        # Perform sampling
        trace_binary = pm.sample(5000, return_inferencedata=True)

        # Summary statistics
        summary = az.summary(trace_binary, var_names=['prior_intervention', 'prior_control'])
        prob_of_benefit = np.mean(trace_binary.posterior['prior_intervention'].values.flatten() < trace_binary.posterior['prior_control'].values.flatten())

        return summary, prob_of_benefit

# Streamlit interface
st.title('Generic Bayesian Analysis Tool')

st.header('Binary Data Analysis')
st.subheader('Example: Mortality, Adverse Events, etc.')
prior_type = st.selectbox('Select Prior Type', ['Neutral', 'Optimistic', 'Pessimistic'])
n_intervention = st.number_input('Number of Patients in Intervention Group', min_value=0, format='%d')
events_intervention = st.number_input('Number of Events in Intervention Group', min_value=0, format='%d')
n_control = st.number_input('Number of Patients in Control Group', min_value=0, format='%d')
events_control = st.number_input('Number of Events in Control Group', min_value=0, format='%d')

if st.button('Analyze Binary Data'):
    summary, prob_of_benefit = bayesian_analysis_binary(n_intervention, events_intervention, n_control, events_control, prior_type)
    st.write(summary)
    st.write(f"Probability of Benefit: {prob_of_benefit}")

# Explanation of Prior Types
st.header('Explanation of Prior Types')
st.markdown("""
- **Neutral Prior**: Assumes that both benefit and harm are equally possible. This is a non-informative prior that doesn't favor any particular outcome.
- **Optimistic Prior**: Leans towards a belief in the intervention's benefit. This prior suggests a higher likelihood of positive outcomes (benefit) over negative ones.
- **Pessimistic Prior**: Leans towards a belief in the intervention's potential harm. This prior suggests a higher likelihood of negative outcomes (harm) over positive ones.
""")

# Explanation of Outputs
st.header('Explanation of Outputs')
st.markdown("""
- **Posterior Mean**: The average value in the posterior distribution, representing our updated belief about the parameter after observing the data.
- **HDI (Highest Density Interval)**: The range within which a certain percentage (e.g., 95%) of the posterior distribution lies. It gives an interval estimate of the parameter.
- **ESS (Effective Sample Size)**: Reflects the number of independent samples in the posterior distribution. Higher ESS indicates more reliable and stable estimates.
- **R-hat**: A measure of convergence for the Markov Chain Monte Carlo (MCMC) sampling. An R-hat close to 1.0 indicates good convergence.
- **Probability of Benefit**: Indicates the likelihood that the intervention group has a better outcome than the control group, based on the data and the selected prior.
""")
