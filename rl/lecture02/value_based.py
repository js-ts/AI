from mdp import MDP
import numpy as np

transition_probs = {
    's0': {
        'a0': {'s0': 0.5, 's2': 0.5},
        'a1': {'s2': 1}
    },
    's1': {
        'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
        'a1': {'s1': 0.95, 's2': 0.05}
    },
    's2': {
        'a0': {'s0': 0.4, 's2': 0.6},
        'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
    }
}
rewards = {
    's1': {'a0': {'s0': +5}},
    's2': {'a1': {'s0': -1}}
}



from mdp import MDP
mdp = MDP(transition_probs, rewards, initial_state='s0')


print(mdp.reset())
print(mdp.step("a0"))
print(mdp.get_all_states())
print(mdp.get_possible_actions("s1"))
print(mdp.get_next_states("s0", "a0"))
print(mdp.get_reward("s0", "a0", "s2"))
print(mdp.get_transition_prob("s0", "a0", "s2"))


def get_action_value(mdp, state_values, state, action, gamma):
    '''
    '''
    next_states = mdp.get_next_states(state, action)

    q = 0
    for s in next_states:
        r = mdp.get_reward(state, action, s)
        p = mdp.get_transition_prob(state, action, s)
        q += p * (r + gamma * state_values[s])

    return q

test_Vs = {s: i for i, s in enumerate(sorted(mdp.get_all_states()))}
assert np.isclose(get_action_value(mdp, test_Vs, 's2', 'a1', 0.9), 0.69)
assert np.isclose(get_action_value(mdp, test_Vs, 's1', 'a0', 0.9), 3.95)
print('assert get_action_value function is correct.')


def get_new_state_value(mdp, state_values, state, gamma):
    '''
    '''
    if mdp.is_terminal(state):
        return 0
    
    actions = mdp.get_possible_actions(state)
    qvalues = [get_action_value(mdp, state_values, state, a, gamma) for a  in actions]

    return max(qvalues)


test_Vs_copy = dict(test_Vs)
assert np.isclose(get_new_state_value(mdp, test_Vs, 's0', 0.9), 1.8)
assert np.isclose(get_new_state_value(mdp, test_Vs, 's2', 0.9), 1.08)
assert test_Vs == test_Vs_copy, "please do not change state_values in get_new_state_value"
print("assert get_new_state_value function is correct.")


gamma = 0.9
num_iter = 100
min_diff = 0.001

state_values = {s: 0 for s in mdp.get_all_states()}

for i in range(num_iter):

    new_state_values = {s: get_new_state_value(mdp, state_values, s, gamma) for s in mdp.get_all_states()}
    assert isinstance(new_state_values, dict)

    diff = max([abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states()])
    state_values = new_state_values

    if diff < min_diff:
        # print('Termineted.')
        break

assert abs(state_values['s0'] - 3.781) < 0.01
assert abs(state_values['s1'] - 7.294) < 0.01
assert abs(state_values['s2'] - 4.202) < 0.01
print('Termineted.', diff, new_state_values)



def get_optimal_action(mdp, state_values, state, gamma=0.9):
    '''
    '''
    if mdp.is_terminal(state):
        return None
    
    actions = mdp.get_possible_actions(state)
    qvalues = [get_action_value(mdp, state_values, state, a, gamma) for a in actions]
    # print(max(qvalues), np.argmax(qvalues))

    return actions[np.argmax(qvalues)] 


assert get_optimal_action(mdp, state_values, 's0', gamma) == 'a1'
assert get_optimal_action(mdp, state_values, 's1', gamma) == 'a0'
assert get_optimal_action(mdp, state_values, 's2', gamma) == 'a1'
print("assert get_optimal_action function is correct.")


s = mdp.reset()
rewards = []
for _ in range(10000):
    s, r, done, _ = mdp.step(get_optimal_action(mdp, state_values, s))
    rewards.append(r)

print(np.mean(rewards))
assert(0.40 < np.mean(rewards) < 0.55)



from mdp import FrozenLakeEnv
mdp = FrozenLakeEnv(slip_chance=0.0)
mdp.render()


def value_iteration(mdp, state_values=None, gamma=0.9, num_iter=1000, min_diff=1e-5):
    '''
    '''
    state_values = state_values or {s: 0 for s in mdp.get_all_states()}

    for _ in range(num_iter):
        new_state_values = {s: get_new_state_value(mdp, state_values, s, gamma) for s in mdp.get_all_states()}
        assert isinstance(new_state_values, dict)
        diff = max([abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states()])
        state_values = new_state_values
        if diff < min_diff:
            break
    
    return state_values

state_values = value_iteration(mdp)

s = mdp.reset()
mdp.render()

for _ in range(100):
    a = get_optimal_action(mdp, state_values, s, gamma=0.9)
    print(a, end="\n\n")
    s, r, done, _ = mdp.step(a)
    mdp.render()
    if done:
        break


# --- 

state_values = {s: 0 for s in mdp.get_all_states()}

for _ in range(300):

    state_values = value_iteration(mdp, state_values, gamma=0.9, num_iter=1)
    policy = {s: get_optimal_action(mdp, state_values, s) for s in mdp.get_all_states()}

print(policy)


# ---

mdp = MDP(transition_probs, rewards, initial_state='s0')