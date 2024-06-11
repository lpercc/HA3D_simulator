import json
import os
import math
import trimesh
import inspect
from tqdm import tqdm
import numpy as np

# Get the path to the HA3D simulator from the environment variable
HA3D_SIMULATOR_PATH = os.environ.get("HA3D_SIMULATOR_PATH")

def printFileAndLineQuick():
    """Quickly print the current file name and line number."""
    line_no = inspect.stack()[1][2]
    file_name = __file__
    print(f"File: {file_name}, Line: {line_no}")

def getHumanInfo(basicDataDir, scanId, agentViewId):
    """
    Determine if a viewpoint can see a person, and if so, return the person's information.

    Parameters:
    - basicDataDir: Path to the base directory containing human information
    - scanId: Scan ID of the building where the viewpoint is located
    - agentViewId: ID of the viewpoint

    Returns:
    - humanHeading: Human orientation in radians
    - humanLoc: Human coordinates [x, y, z]
    - motionPath: Path to the 3D mesh data of the human motion
    """
    motionDir = os.path.join(basicDataDir, "human_motion_meshes")
    
    # Load human viewpoint annotation data
    with open(os.path.join(HA3D_SIMULATOR_PATH, 'human-viewpoint_annotation/human_motion_text.json'), 'r') as f:
        humanViewData = json.load(f)
    
    # Load position information of the scan
    with open(os.path.join(HA3D_SIMULATOR_PATH, f'con/pos_info/{scanId}_pos_info.json'), 'r') as f:
        posData = json.load(f)
    
    # Load connection information of the scan
    with open(os.path.join(HA3D_SIMULATOR_PATH, f'con/con_info/{scanId}_con_info.json'), 'r') as f:
        connectionData = json.load(f)

    humanHeading, humanLoc, motionPath = None, None, None
    
    for humanViewId in humanViewData[scanId]:
        humanMotion = humanViewData[scanId][humanViewId][0]
        humanModelId = humanViewData[scanId][humanViewId][1]
        try:
            if humanViewId == agentViewId:
                connectionData[agentViewId]["visible"].append(agentViewId)
            
            if humanViewId in connectionData[agentViewId]['visible']:
                motionPath = os.path.join(motionDir, humanMotion.replace(' ', '_').replace('/', '_'), f"{humanModelId}_obj")
                humanLoc = [posData[humanViewId][0], posData[humanViewId][1], posData[humanViewId][2]]
                humanHeading = humanViewData[scanId][humanViewId][2]
        except KeyError:
            pass

    return humanHeading, humanLoc, motionPath

def getHumanOfScan(scanId):
    """Get human mesh data of a building."""
    humanList = []
    motionDir = os.path.join(os.environ.get("HA3D_SIMULATOR_DATA_PATH"), "human_motion_meshes")
    
    with open(os.path.join(HA3D_SIMULATOR_PATH, 'human-viewpoint_annotation/human_motion_text.json'), 'r') as f:
        humanViewData = json.load(f)
    
    with open(os.path.join(HA3D_SIMULATOR_PATH, f'con/pos_info/{scanId}_pos_info.json'), 'r') as f:
        posData = json.load(f)
    
    with open(os.path.join(HA3D_SIMULATOR_PATH, f'con/con_info/{scanId}_con_info.json'), 'r') as f:
        connectionData = json.load(f)
    
    for humanViewId in humanViewData[scanId]:
        humanMeshes = []
        humanMotion = humanViewData[scanId][humanViewId][0]
        humanModelId = humanViewData[scanId][humanViewId][1]
        humanHeading = humanViewData[scanId][humanViewId][2]
        humanLoc = [posData[humanViewId][0], posData[humanViewId][1], posData[humanViewId][2]]
        motionPath = os.path.join(motionDir, humanMotion.replace(' ', '_').replace('/', '_'), f"{humanModelId}_obj")
        
        objFiles = sorted([f for f in os.listdir(motionPath) if f.endswith('.obj')])
        
        for objFile in objFiles[:120]:
            objPath = os.path.join(motionPath, objFile)
            mesh = trimesh.load(objPath)
            humanMeshes.append(mesh)
        
        humanList.append({
            'heading': humanHeading,
            'location': humanLoc,
            'meshes': humanMeshes
        })
    return humanList

def getRotation(theta=np.pi):
    """Get the rotation matrix for a given angle around the Y-axis."""
    import src.utils.rotation_conversions as geometry
    import torch
    
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisAngle = theta * axis
    matrix = geometry.axisAngleToMatrix(axisAngle)
    return matrix.numpy()

def getHumanLocations(scanId):
    """Get locations of humans in a given scan."""
    humanLocations = []
    humanList = getHumanOfScan(scanId)
    
    for human in humanList:
        location = human['location']
        aHumanLocation = []
        humanStartLoc = (location[0], location[2] - 1.36, -location[1])
        thetaAngle = (np.pi / 180 * float(human['heading']))
        matrix = getRotation(theta=thetaAngle)
        minDistance = 1
        originIndex = 0
        
        for index, item in enumerate(human['meshes'][0].vertices):
            distance = np.sum(np.square(item))
            if distance < minDistance:
                minDistance = distance
                originIndex = index
        
        for mesh in human['meshes']:
            mesh.vertices = np.einsum("ij,ki->kj", matrix, mesh.vertices)
            mesh.vertices += humanStartLoc
            meshLocation = mesh.vertices[originIndex]
            aHumanLocation.append((meshLocation[0], -meshLocation[2], meshLocation[1] + 1.36))
        
        humanLocations.append(aHumanLocation)
    return humanLocations

def getAllHumanLocations(scanIds=[]):
    """Get all human locations for each building."""
    filePath = os.path.join(HA3D_SIMULATOR_PATH, "tasks/HA/data/human_locations.json")
    
    if os.path.exists(filePath):
        with open(filePath, 'r') as j:
            data = json.load(j)
            if len(data) == 90:
                return data
    
    allHumanLocations = {}
    
    with open(os.path.join(HA3D_SIMULATOR_PATH, "connectivity/scans.txt")) as f:
        scans = scanIds if scanIds else [scan.strip() for scan in f.readlines()]
        
        for scan in tqdm(scans, desc='Loading Human meshes'):
            humanLocationsOfScan = getHumanLocations(scan)
            allHumanLocations[scan] = humanLocationsOfScan
    
    with open(filePath, 'w') as j:
        json.dump(allHumanLocations, j, indent=4)
    
    return allHumanLocations

def getHumanOnPath(dataDirPath):
    """Calculate the visible humans on each path in the dataset."""
    print(f"Processing: {dataDirPath}")
    
    with open(os.path.join(HA3D_SIMULATOR_PATH, 'human-viewpoint_annotation/human_motion_text.json'), 'r') as f:
        humanViewData = json.load(f)
    
    r2rData = readVlnData(dataDirPath)
    newR2rData = []

    allPathNum = 0
    beginningPathNum = 0
    obstaclePathNum = 0
    aroundPathNum = 0
    endPathNum = 0

    humanNum = 0
    beginningNum = 0
    obstacleNum = 0
    aroundNum = 0
    endNum = 0

    for r2rDataItem in r2rData:
        humanInfo = []
        scanId = r2rDataItem["scan"]
        path = r2rDataItem["path"]
        pathId = r2rDataItem["path_id"]
        
        with open(os.path.join(HA3D_SIMULATOR_PATH, f'con/pos_info/{scanId}_pos_info.json'), 'r') as f:
            posData = json.load(f)
        
        with open(os.path.join(HA3D_SIMULATOR_PATH, f'con/con_info/{scanId}_con_info.json'), 'r') as f:
            connectionData = json.load(f)
        
        pathVisiblePoints = getVisiblePoints(path, connectionData)
        
        for visiblePoint in pathVisiblePoints:
            if visiblePoint in humanViewData[scanId]:
                humanRelPos = getRelPos(visiblePoint, path, pathId, posData)
                humanNum += 1
                beginningNum += int(humanRelPos == "Beginning")
                obstacleNum += int(humanRelPos == "Obstacle")
                aroundNum += int(humanRelPos == "Around")
                endNum += int(humanRelPos == "End")
                humanInfo.append({
                    "human_viewpoint": visiblePoint,
                    "human_rel_pos": humanRelPos,
                    "human_description": humanViewData[scanId][visiblePoint][0]
                })
        
        if humanInfo:
            allPathNum += 1
            beginningFlag = obstacleFlag = aroundFlag = endFlag = 0
            
            for item in humanInfo:
                beginningFlag |= int(item["human_rel_pos"] == "Beginning")
                obstacleFlag |= int(item["human_rel_pos"] == "Obstacle")
                aroundFlag |= int(item["human_rel_pos"] == "Around")
                endFlag |= int(item["human_rel_pos"] == "End")
            
            beginningPathNum += beginningFlag
            obstaclePathNum += obstacleFlag
            aroundPathNum += aroundFlag
            endPathNum += endFlag

        r2rDataItem["human"] = humanInfo
        newR2rData.append(r2rDataItem)
    
    print(f"Paths with humans: {allPathNum} / Total paths: {len(newR2rData)}")
    print("Number of paths containing each relative position:")
    print(f"All paths containing relative positions: {allPathNum}")
    print(f"Beginning: {beginningPathNum}")
    print(f"Obstacle: {obstaclePathNum}")
    print(f"Around: {aroundPathNum}")
    print(f"End: {endPathNum}")
    print("Number of relative positions of each human:")
    print(f"All relative positions: {humanNum}")
    print(f"Beginning: {beginningNum}")
    print(f"Obstacle: {obstacleNum}")
    print(f"Around: {aroundNum}")
    print(f"End: {endNum}")

    with open(f"{dataDirPath.split('.json')[0]}_human.json", 'w') as f:
        json.dump(newR2rData, f, indent=4)

def getRelPos(humanPoint, path, pathId, posData):
    """Calculate the relative position of a human on the path."""
    locDesc = ["Beginning", "Obstacle", "Around", "End"]
    minDistance = float('inf')
    
    for index, pathPoint in enumerate(path):
        distance = computeDistance(humanPoint, pathPoint, posData)
        if distance < minDistance:
            minDistance = distance
            if distance < 1.5:
                if index == 0:
                    humanRelPos = locDesc[0]
                elif index == len(path) - 1:
                    humanRelPos = locDesc[-1]
                else:
                    humanRelPos = locDesc[1]
            else:
                humanRelPos = locDesc[2]
    return humanRelPos

def relHumanAngle(humanLocations, agentLocation, agentHeading, agentElevation):
    """Calculate the relative heading and elevation angles to the nearest human."""
    nearestHuman = []
    minDistance = float('inf')
    
    for humanLocation in humanLocations:
        distance = np.linalg.norm(np.array(humanLocation) - np.array(agentLocation))
        if distance < minDistance:
            minDistance = distance
            nearestHuman = humanLocation
    
    headingAngle, elevationAngle = horizontalAndElevationAngles(agentLocation, nearestHuman)
    return headingAngle - agentHeading, elevationAngle - agentElevation, minDistance

def horizontalAndElevationAngles(point1, point2):
    """
    Calculate the relative horizontal angle and elevation angle between two 3D coordinates.
    
    Parameters:
    - point1: First 3D coordinate
    - point2: Second 3D coordinate
    
    Returns:
    - Horizontal angle and elevation angle in radians
    """
    vector = np.array(point2) - np.array(point1)
    horizontalAngle = np.arctan2(vector[0], vector[1])
    elevationAngle = np.arctan2(vector[2], np.linalg.norm(vector[:2]))
    return horizontalAngle, elevationAngle

def loadViewpointIds():
    """Load all viewpoint IDs."""
    graphsPath = "connectivity/"
    viewpointIds = []
    
    with open(os.path.join(HA3D_SIMULATOR_PATH, graphsPath + "scans.txt")) as f:
        scans = [scan.strip() for scan in f.readlines()]
        
        for scan in scans:
            with open(graphsPath + scan + "_connectivity.json") as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpointIds.append((scan, item["image_id"]))
    
    print(f"Loaded {len(viewpointIds)} viewpoints")
    return viewpointIds

def computeDistance(viewpointId1, viewpointId2, posData):
    """Compute the distance between two viewpoints."""
    xDis = posData[viewpointId1][0] - posData[viewpointId2][0]
    yDis = posData[viewpointId1][1] - posData[viewpointId2][1]
    zDis = posData[viewpointId1][2] - posData[viewpointId2][2]
    squaredSum = xDis**2 + yDis**2 + zDis**2
    return math.sqrt(squaredSum)

def readVlnData(filePath):
    """Read VLN data from a file."""
    with open(filePath, 'r') as f:
        r2rData = json.load(f)
    
    print("R2R dataset:")
    print(f"Total paths: {len(r2rData)}")
    print(f"Total instructions: {len(r2rData) * 3}")
    
    return r2rData

def getVisiblePoints(path, connectionData):
    """Get visible points around a path (including the path points)."""
    pathVisiblePoints = []
    
    try:
        for pathPoint in path:
            visiblePoints = connectionData[pathPoint]['visible']
            for point in visiblePoints:
                if point not in pathVisiblePoints:
                    pathVisiblePoints.append(point)
    except KeyError:
        pass
    
    return pathVisiblePoints

def getUnobstructedPoints(path, connectionData):
    """Get unobstructed points around a path."""
    pathUnobstructedPoints = []
    
    try:
        for pathPoint in path:
            unobstructedPoints = connectionData[pathPoint]['unobstructed']
            for point in unobstructedPoints:
                if point not in pathUnobstructedPoints:
                    pathUnobstructedPoints.append(point)
    except KeyError:
        pass
    
    return pathUnobstructedPoints

def countPointsSeenHuman():
    """Count the number of viewpoints that can see humans."""
    allViewpointIds = loadViewpointIds()
    viewpointsCounts = len(allViewpointIds)
    humanVisibleCounts = 0
    
    for _, (scanId, viewpointId) in enumerate(allViewpointIds):
        _, humanLoc, _ = getHumanInfo("./", scanId, viewpointId)
        if humanLoc is not None:
            humanVisibleCounts += 1
    
    print(f"Human visible points: {humanVisibleCounts} / All points: {viewpointsCounts}")

def countCommonElements(list1, list2):
    """Count the number of common elements between two lists."""
    set1 = set(list1)
    set2 = set(list2)
    commonElements = set1 & set2
    return len(commonElements)

def getCruxOnPath(dataFile):
    """Get the key points on the path that must be passed to reach the goal."""
    data = readVlnData(dataFile)
    
    for j, dataItem in enumerate(data):
        scanId = dataItem["scan"]
        
        with open(os.path.join(HA3D_SIMULATOR_PATH, f'con/con_info/{scanId}_con_info.json'), 'r') as f:
            connectionData = json.load(f)
        
        cruxList = [dataItem["path"][0]]
        
        for i, viewpoint in enumerate(dataItem["path"]):
            if len(dataItem["path"]) < 2:
                break
            nextViewpoint = dataItem["path"][i + 1]
            if nextViewpoint == dataItem["path"][-1]:
                cruxList.append(nextViewpoint)
                break
            
            unobstructedPoints = connectionData[viewpoint]['unobstructed']
            nextUnobstructedPoints = connectionData[nextViewpoint]['unobstructed']
            
            if countCommonElements(unobstructedPoints, nextUnobstructedPoints) == 1:
                cruxList.append(nextViewpoint)
        
        data[j]["crux_points"] = cruxList
    
    print(dataFile)
    
    with open(f"{dataFile.split('.')[0]}_crux_.json", 'w') as f:
        json.dump(data, f, indent=4)

def countHumanOfRegion():
    """Count the number of humans in each region."""
    region = {}
    
    with open(os.path.join(HA3D_SIMULATOR_PATH, 'human-viewpoint_annotation/human_motion_text.json'), 'r') as f:
        humanViewData = json.load(f)
    
    for i, scanId in enumerate(humanViewData):
        print(f"{i}th scan {scanId}")
        for humanViewpointId in humanViewData[scanId]:
            print(f"**Human viewpoint {humanViewpointId}")
            humanRegion = humanViewData[scanId][humanViewpointId][0].split(':')[0]
            region[humanRegion] = region.get(humanRegion, 0) + 1
    
    print(region)
    print(len(region))

def forwardViewpointIdx(navigableLocations):
    """Calculate the next viewpoint index for the agent to move forward."""
    fieldAngle = math.radians(10)
    minDistance = 10
    nextViewpointIdx = 0
    
    for idx, loc in enumerate(navigableLocations[1:]):
        if abs(loc.rel_heading) <= fieldAngle and abs(loc.rel_elevation) <= fieldAngle and loc.rel_distance < minDistance:
            minDistance = loc.rel_distance
            nextViewpointIdx = idx + 1
    
    return nextViewpointIdx

def readScanIdList(filePath):
    """Read a list of scan IDs from a file."""
    with open(filePath) as f:
        scanIdList = [scan.strip() for scan in f.readlines()]
    return scanIdList

def readConnectionData(filePath):
    """Read connection data from a file."""
    with open(filePath, 'r') as f:
        connectionData = json.load(f)
    return connectionData

def readPositionData(filePath):
    """Read position data from a file."""
    with open(filePath, 'r') as f:
        positionData = json.load(f)
    return positionData

if __name__ == '__main__':
    # Example usage
    dataFolder = os.path.join(HA3D_SIMULATOR_PATH, 'tasks/R2R/data')
    
    # Uncomment the lines below to run specific functions
    # files = [f for f in os.listdir(dataFolder) if f.endswith('.json')]
    # for file in files:
    #     getCruxOnPath(os.path.join(dataFolder, file))
    # getHumanOnPath(os.path.join(dataFolder, "R2R_train.json"))
    # getHumanOnPath(os.path.join(dataFolder, "R2R_val_seen.json"))
    # getHumanOnPath(os.path.join(dataFolder, "R2R_val_unseen.json"))
    # getHumanOnPath(os.path.join(dataFolder, "path.json"))
    countHumanOfRegion()
    # countPointsSeenHuman()
