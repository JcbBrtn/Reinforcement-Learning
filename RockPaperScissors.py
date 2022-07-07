"""
Rock, Paper, Scissors! The basic game where Rock=1, Paper=2, Scissors=3
"""

def play(agent1, agent2, rounds=300):
    agent1_move = None
    agent2_move = None
    score1, score2 = 0, 0
    for round_num in range(rounds):
        agent1_obs = {
            'Opponent' : agent2_move,
            'Round' : round_num
            }
        agent2_obs = {
            'Opponent' : agent1_move,
            'Round' : round_num
            }
        agent1_move = agent1.run(agent1_obs)
        agent2_move = agent2.run(agent2_obs)

        score1, score2 = get_scores(agent1_move, agent2_move, score1, score2)

    if score1 > score2:
        return agent1
    if score1 < score2:
        return agent2
    else:
        return 0

def get_scores(agent1_move, agent2_move, score1, score2):
    if agent1_move == agent2_move:
        #Tie!
        return score1, score2
    elif (agent1_move % 3) + 1 == agent2_move:
        #Agent1 won that round
        return score1+1, score2
    else:
        return score1, score2+1