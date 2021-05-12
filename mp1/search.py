# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
import heapq

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    dict = {maze.start : maze.start}
    queue = [maze.start]
    s = queue.pop(0)
    while s != maze.waypoints[0]:
        neigh = maze.neighbors(s[0],s[1])
        for i in neigh:
            if i not in dict:
                dict[i] = s
                queue.append(i)
        s = queue.pop(0)
    path = []
    while s != maze.start:
        path.insert(0, s)
        s = dict[s]
    path.insert(0, s)
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    dict = {maze.start : (maze.start, 0)}
    heap = []
    heapq.heappush(heap, (abs(maze.start[0] - maze.waypoints[0][0]) + abs(maze.start[1] - maze.waypoints[0][1]), 0, maze.start))
    s = heapq.heappop(heap)
    while s[2] != maze.waypoints[0]:
        neigh = maze.neighbors(s[2][0],s[2][1])
        for i in neigh:
            if i not in dict or dict[i][1] > s[1] + 1:
                dict[i] = (s, s[1] + 1)
                dist = abs(i[0] - maze.waypoints[0][0]) + abs(i[1] - maze.waypoints[0][1])
                heapq.heappush(heap, (dist + s[1] + 1, s[1] + 1, i))
        s = heapq.heappop(heap)
    path = []
    while s[2] != maze.start:
        path.insert(0, s[2])
        s = dict[s[2]][0]
    path.insert(0, s[2])
    return path

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    dict = {}
    paths = {}
    heap = []
    edge = []
    for i in range(len(maze.waypoints)):
        for j in range(i + 1, len(maze.waypoints)):
            heapq.heappush(edge, (abs(maze.waypoints[i][0] - maze.waypoints[j][0]) + abs(maze.waypoints[i][1] - maze.waypoints[j][1]), maze.waypoints[i], maze.waypoints[j]))
    myset = set()
    sum = 0
    while len(myset) != len(maze.waypoints):
        s = heapq.heappop(edge)
        if s[1] not in myset or s[2] not in myset:
            myset.add(s[1])
            myset.add(s[2])
            sum += s[0]
    mst = {maze.waypoints : sum}
    for i in range(len(maze.waypoints)):
        unexplored = list(maze.waypoints)
        point = maze.waypoints[i]
        unexplored.remove(point)
        waypoints = maze.waypoints
        maze.waypoints = [point]
        path = astar_single(maze)
        paths[(maze.start, point)] = path
        dist = len(path)
        maze.waypoints = waypoints
        dict[(point, tuple(unexplored))] = dist
        heapq.heappush(heap, (sum + dist, dist, [maze.start, point], unexplored))
    pt = heapq.heappop(heap)
    while len(pt[3]) > 0:
        if len(pt[3]) == 1:
            mylist = pt[2].copy()
            if (mylist[len(mylist) - 1], pt[3][0]) in paths:
                dist = len(paths[(mylist[len(mylist) - 1], pt[3][0])])
            else:
                start = maze.start
                waypoints = maze.waypoints
                maze.start = mylist[len(mylist) - 1]
                maze.waypoints = [pt[3][0]]
                path = astar_single(maze)
                paths[(maze.start, pt[3][0])] = path
                paths[(pt[3][0], maze.start)] = path[::-1]
                dist = len(path)
                maze.start = start
                maze.waypoints = waypoints
            mylist.append(pt[3][0])
            if (pt[3][0], ()) not in dict or dict[(pt[3][0], ())] > dist + pt[1]:
                dict[(pt[3][0], ())] = dist + pt[1]
                heapq.heappush(heap, (dist + pt[1], dist + pt[1], mylist, []))
            pt = heapq.heappop(heap)
            continue
        for i in range(len(pt[3])):
            unexplored = pt[3].copy()
            if tuple(unexplored) not in mst:
                edges = []
                for j in range(len(unexplored)):
                    for k in range(j + 1, len(unexplored)):
                        heapq.heappush(edges, (abs(unexplored[j][0] - unexplored[k][0]) + abs(unexplored[j][1] - unexplored[k][1]), unexplored[j], unexplored[k]))
                myset = set()
                sum = 0
                while len(myset) != len(unexplored):
                    s = heapq.heappop(edges)
                    if s[1] not in myset or s[2] not in myset:
                        myset.add(s[1])
                        myset.add(s[2])
                        sum += s[0]
                mst[tuple(unexplored)] = sum
            sum = mst[tuple(unexplored)]
            point = pt[3][i]
            unexplored.remove(point)
            mylist = pt[2].copy()
            if (mylist[len(mylist) - 1], point) in paths:
                dist = len(paths[(mylist[len(mylist) - 1], point)])
            else:
                start = maze.start
                waypoints = maze.waypoints
                maze.start = mylist[len(mylist) - 1]
                maze.waypoints = [point]
                path = astar_single(maze)
                paths[(maze.start, point)] = path
                paths[(point, maze.start)] = path[::-1]
                dist = len(path)
                maze.start = start
                maze.waypoints = waypoints
            mylist.append(pt[3][i])
            if (point, tuple(unexplored)) not in dict or dict[(point, tuple(unexplored))] > dist + pt[1]:
                dict[(point, tuple(unexplored))] = dist + pt[1]
                heapq.heappush(heap, (sum + dist + pt[1], dist + pt[1], mylist, unexplored))
        pt = heapq.heappop(heap)
    path = [maze.start]
    n = 0
    while n < len(pt[2]) - 1:
        path.extend(paths[(pt[2][n], pt[2][n + 1])][1:])
        n += 1
    return path

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return astar_corner(maze)

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    
            
