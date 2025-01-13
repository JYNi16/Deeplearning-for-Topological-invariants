import numpy as np
import os

def mk_path(save_path):
    if os.path.exists(save_path):
        print("save path {} exist".format(save_path))
    else:
        print("save path {} not exist".format(save_path))
        os.makedirs(save_path)
        print("now makedir the save_path")


save_path = "train_data1"
save_val_path = "val_test1"

mk_path(save_path)
mk_path(save_val_path)


class SSHmodel():
    def __init__(self):
        self.numk = 33
        self.kk = np.linspace(0, 2 * np.pi, num=self.numk)
        self.kk[-1] = self.kk[0]

        self.n_train = 40000
        self.n_test = 20000

        self.n_casos = int(self.n_train + self.n_test)

    def isLeft(self, P0, P1, P2):
        """
        isLeft(): tests if a point is Left|On|Right of an infinite line.
        Input :  three points P0, P1, and P2
        Return: >0 for P2 left of the line through P0 and P1
                =0 for P2  on the line
                <0 for P2  right of the line
        """
        return ((P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1]))

    def wn_PnPoly(self, P, V):
        """
        wn_PnPoly(): winding number for a point in a polygon
        Input:   P = a point, V = vertex points of a polygon V[n+1] with V[n]=V[0]
        Return:  wn = the winding number (=0 only when P is outside)
        """
        wn = 0;  # the  winding number counter

        # loop through all edges of the polygon
        for i in range(len(V) - 1):  # edge from V[i] to  V[i+1]  for(int i=0; i<n; i++)
            if (V[i][1] <= P[1]):  # start y <= P.y
                if (V[i + 1][1] > P[1]):  # an upward crossing
                    if (self.isLeft(V[i], V[i + 1], P) > 0):  # P left of  edge
                        wn += 1;  # have  a valid up intersect

            else:  # start y > P.y (no test needed)
                if (V[i + 1][1] <= P[1]):  # a downward crossing
                    if (self.isLeft(V[i], V[i + 1], P) < 0):  # P right of  edge
                        wn -= 1;  # have  a valid down intersect

        return wn

    def sampledata(self):

        for i in range(self.n_casos):
            # "Hamiltonian parameters"
            max_c = 1
            axx = (np.random.random(max_c + 1) * 2) - 1
            bxx = (np.random.random(max_c + 1) * 2) - 1
            ayy = (np.random.random(max_c + 1) * 2) - 1
            byy = (np.random.random(max_c + 1) * 2) - 1
            ##
            hx_list = []
            hy_list = []

            for k in (self.kk):
                hnx = np.sum([ax * np.cos(i * k) + bx * np.sin(i * k) for i, (ax, bx) in enumerate(zip(axx, bxx))])
                hny = np.sum([ay * np.cos(i * k) + by * np.sin(i * k) for i, (ay, by) in enumerate(zip(ayy, byy))])

                hx = hnx / np.sqrt(hnx * hnx + hny * hny)
                hy = hny / np.sqrt(hnx * hnx + hny * hny)

                hx_list.append(hx)
                hy_list.append(hy)

            V = np.array([hx_list, hy_list]).T
            P = (0, 0)
            wn = self.wn_PnPoly(P, V)
            
            
            print("winding number is:", wn)

            if i % 3 == 0:
                np.savez(save_val_path + "/{}.npz".format(i), s=V, label = wn)
            else:
                np.savez(save_path + "/{}.npz".format(i), s=V, label = wn)



if __name__=="__main__":
    data = SSHmodel()
    data.sampledata()


