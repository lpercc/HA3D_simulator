from utils import load_datasets, load_nav_graphs, relHumanAngle

agentLocation = [-6.99,1.40,1.52]
humanLocations = [[-6.42,1.38,0]]
point3 = [1,1,0]

relHeading, relElevation = relHumanAngle(humanLocations, agentLocation)
print(relHeading, relElevation)