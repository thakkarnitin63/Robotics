
import matplotlib.pyplot as plot
from PIL import Image
import numpy as np
global occupancy_grid
occupancy_map_img=Image.open(r"E:\NEU MAIN\Mobile Robotics\Mobile Assignment\occupancy_map.png")
occupancy_grid=(np.asarray(occupancy_map_img) > 0).astype(int)



Vertex={}
CostTo={}
pred={}
EstTotalCost={}
Prio_Queue=[]

print(occupancy_grid.shape)
for row in range(occupancy_grid.shape[0]):
    for col in range(occupancy_grid.shape[1]):
        Vertex[(row,col)] = occupancy_grid[row][col]



def edge_weight(w1,w2):
    xz1,yz1 = w1
    xz2,yz2 = w2
    resultant_edge = np.sqrt(((yz2 - yz1) ** 2) + ((xz2 - xz1) ** 2))
    return resultant_edge




def h_values(v1,v2):
    x1,y1=v1
    x2,y2=v2
    resultant_h=np.sqrt(((y2-y1)**2)+((x2-x1)**2))
    return resultant_h


def Neighbor(v,vertex_list):
    x, y = v
    actual_neighbor = []

    neighbor_check = [(x - 1, y + 1), (x, y + 1), (x + 1, y + 1), (x - 1, y), (x + 1, y), (x - 1, y - 1), (x, y - 1),
                      (x + 1, y - 1)]

    for nei in neighbor_check:
        if vertex_list[nei]==1:
            actual_neighbor.append(nei)
    return actual_neighbor

def RecoverPath(start,goal,pred):
    reached=[goal]
    r=pred[goal]
    distance=edge_weight(goal,r)
    while r != start:
        distance+=edge_weight(r,pred[r])
        reached.append(r)
        r=pred[r]
    reached.append(start)
    return reached,distance







def A_star_Search(Vertex,start,goal):
    for v in Vertex:
        CostTo[v]=float("inf")
        EstTotalCost[v]=float("inf")



    CostTo[start]=0
    EstTotalCost[start]=h_values(start,goal)
    Prio_Queue=[(h_values(start,goal),start)]
    while (len(Prio_Queue)>0):
        v=Prio_Queue.pop(0)
        if v[1]==goal:
            return RecoverPath(start,goal,pred)


        neighbour_list = Neighbor(v[1],Vertex)
        for i in neighbour_list:
            pvi=CostTo[v[1]]+edge_weight(v[1],i)
            if pvi<CostTo[i]:
                pred[i]=v[1]
                if (EstTotalCost[i],i) in Prio_Queue:
                    Prio_Queue.remove((EstTotalCost[i],i))
                CostTo[i]=pvi
                EstTotalCost[i]=pvi+h_values(i,goal)
                Prio_Queue.append((EstTotalCost[i],i))
                Prio_Queue.sort()


p,distance = A_star_Search(Vertex,(635,140),(350,400))
p = np.array(p)
print(distance)

plot.scatter(p[:,1],p[:,0], s=1)
plot.imshow(occupancy_grid, cmap ='gray')
plot.show()
