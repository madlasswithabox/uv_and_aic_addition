from sage.all import *
import copy

class Vertex:
    
    """
    Signifies a copy of R.

    attributes:
    * grading: a tuple of ints (i,j) for a what (u,v) grading 1 is in
    * left, right, up, down, each of which is an edge, could be None
    * position: either a tuple of ints (a,b) if vertex of grid, otherwise int a
    * incoming_edges: as a list
    * outgoing_edges: as a list

    """

    def __init__(self):
        self.left = None
        self.right = None
        self.up = None
        self.down = None

    def set_in_out(self):
        self.incoming_edges = []
        self.outgoing_edges = []

        if self.left != None:
            if self.left.direction == "left":
                self.outgoing_edges.append(self.left)

            else:
                self.incoming_edges.append(self.left)
        if self.right != None:
            if self.right.direction == "right":
                self.outgoing_edges.append(self.right)

            else:
                self.incoming_edges.append(self.right)

        if self.up != None:
            if self.up.direction == "up":
                self.outgoing_edges.append(self.up)

            else:
                self.incoming_edges.append(self.up)

        if self.down != None:
            if self.down.direction == "down":
                self.outgoing_edges.append(self.down)

            else:
                self.incoming_edges.append(self.down)


                

class Edge:
    """ 
    Signifies an arrow in a grid or a chain
    
    attributes:
    * v1: a vertex
    * v2: a vertex
    * direction: one of "left", "right", "up", "down"
    * value: (i,j) where either i or j is 0, and it's u^i or v^j
    """ 

    def opposite(self, v):
        """ 
        Take a vertex v and returns the position of the 
        vertex at the opposite end.
        """

        if self.v1.position == v.position:
            return self.v2.position
        else:
            return self.v1.position


class Singleton:
    """
    A single element of the grid

    attributes: 
    * position: a tuple of ints (a,b)
    * value: (i,j) where at least one of i or j is 0. Means u^i v^j

    """


class Element:
    """
    An element of the chain group, recorded as a tuple of singletons
    """



    
def make_gradings_chain(chain):
    """
    seq is an array of vertices, which do not have gradings set,
    we will set the gradings.
    """
    n = len(chain)

    chain[0].grading = (0,0)
    for t in range(1,n):
        v = chain[t]
        (prev0,prev1) = chain[t-1].grading
        e = v.left
        (i,j) = e.value
        if e.direction == "left":
            im0 = prev0 -2*i
            im1 = prev1 -2*j
            # These are the gradings of the image of 1, (this image is in previous)
            v.grading = (im0+1, im1+1)

        elif e.direction == "right":
            im0 = prev0 -1
            im1 = prev1 -1
            # These are the gradings of the image of 1 (this image is in current)
            v.grading = (im0+2*i, im1+2*j)


    


def make_gradings_grid(grid):
    """
    Making the gradings, assuming that the gradings on the 0'th row is already correct, 
    which it should be if you have down 
    """
    for t in range(1,len(grid)):
        for t1 in range(len(grid[t])):
            v = grid[t][t1]
            (prev0, prev1) = grid[t-1][t1].grading # this the one below
            e = v.down
            
            (i,j) = e.value
            if e.direction == "down":
                im0 = prev0 -2*i
                im1 = prev1 -2*j
                # These are the gradings of the image of 1, (this image is in previous)
                v.grading = (im0+1, im1+1)

            elif e.direction == "up":
                im0 = prev0 -1
                im1 = prev1 -1
                # These are the gradings of the image of 1 (this image is in current)
                v.grading = (im0+2*i, im1+2*j)

    
def make_chain(seq):
    v1 = Vertex()
    v1.position = 0

    
    chain = [v1]
    for i in range(len(seq)):
        edge = Edge()
        v2 = Vertex()
        v2.position = i+1
        val = 0 # stores the absolute value of seq[i]
        if seq[i]>0:
            edge.direction = "left"
            val = seq[i]
        else:
            edge.direction = "right"
            val = -seq[i]
            
        edge.v1 = v1
        edge.v2 = v2
        if (i % 2) == 0: # So it's a u map
            edge.value = (val,0)
        else:
            edge.value = (0, val)

        v1.right = edge
        v2.left = edge
        
        chain.append(v2)
        v1 = v2
        
    make_gradings_chain(chain)
    for v in chain:
        v.set_in_out()

    return chain


def make_grid(chain1, chain2):
    grid = [] # each item is a copy of chain1, up is going up in coordinate
    # So grid[t+1] is above grid[t]
    for t in range(len(chain2)):
        grid.append(copy.deepcopy(chain1))
        for t1 in range(len(grid[t])):
            grid[t][t1].position = (t,t1)
    for t in range(len(chain2) - 1):
        for t1 in range(len(chain1)):
            edge  = Edge()
            edge.v1 = grid[t][t1]
            edge.v2 = grid[t+1][t1]
            if chain2[t].right.direction == "right":
                edge.direction = "up"
            else:
                edge.direction = "down"

            edge.value = chain2[t].right.value
            grid[t][t1].up = edge
            grid[t+1][t1].down = edge
    for row in grid:
        for v in row:
            v.set_in_out()
            
    make_gradings_grid(grid)
    return grid



def check_chain(chain):
    for i in range(len(chain)-1):
        print "Grading: " + str(chain[i].grading)
        print "Right direction: " + chain[i].right.direction
        print (chain[i+1].left == chain[i].right)
        print "Right value: " + str(chain[i].right.value)

    print "Grading: " + str(chain[-1].grading)
        
            
def check_grid(grid):
    print len(grid)
    for row in grid:
        for vertex in row:
            print "Vertex's edges values:"
            edges = [vertex.up, vertex.down, vertex.left, vertex.right]
            print vertex.incoming_edges
            edge_values = []
            for edge in edges:
                if edge != None:
                    edge_values.append(edge.value)

                else:
                    edge_values.append(None)

            print edge_values


def find_grading_basis(grid, grading):
    """
    basis elements are ((t1,t2), (m,n)) where this is u^mv^n in position (t1,t2)
    in order of row, then column
    """
    basis = []
    (i,j) = grading
    for row in grid:
        for v in row:
            (i1,j1) = v.grading
            if i == i1 and j <= j1:
                basis.append( (v.position, (0,(j1-j)/2) ) )
            elif j == j1 and i <= i1:
                basis.append( (v.position, ((i1-i)/2,0) ) )

                
    return basis


def find_boundary_map(grid, grading_bases, grading):
    (i,j) = grading
    if (i,j) not in grading_bases:
        grading_bases[(i,j)] = find_grading_basis(grid, (i,j))
    if (i-1,j-1) not in grading_bases:
        grading_bases[(i-1, j-1)] = find_grading_basis(grid, (i-1,j-1))

    n = len(grading_bases[(i,j)])
    m = len(grading_bases[(i-1,j-1)])
    #R = IntegerModRing(2)
    R = GF(2)
    M_entries = [[0 for x in range(m)] for y in range(n)]
    for source_index in range(n):
        ((a,b), (expu, expv)) = grading_bases[(i,j)][source_index]
        for edge in grid[a][b].outgoing_edges:
            (c,d) = edge.opposite(grid[a][b])
            (edgeval1, edgeval2) = edge.value
            e1 = edgeval1 + expu
            e2 = edgeval2 + expv

            if e1>0 and e2 >0:
                continue


            # Otherwise ((c,d),(e1,e2)) should be in grading_bases(i-1,j-1)
            target_index = grading_bases[(i-1,j-1)].index(((c,d),(e1,e2)))
            M_entries[source_index][target_index] += 1
            
    M = Matrix(R, M_entries)
    return M


def find_boundary_map_u(grid, grading_bases, grading):
    (i,j) = grading
    if (i,j) not in grading_bases:
        grading_bases[(i,j)] = find_grading_basis(grid, (i,j))
    if (i-1,j-1) not in grading_bases:
        grading_bases[(i-1, j-1)] = find_grading_basis(grid, (i-1,j-1))

    n = len(grading_bases[(i,j)])
    m = len(grading_bases[(i-1,j-1)])
    #R = IntegerModRing(2)
    R = GF(2)
    M_entries = [[0 for x in range(m)] for y in range(n)]
    for source_index in range(n):
        ((a,b), (expu, expv)) = grading_bases[(i,j)][source_index]
        for edge in grid[a][b].outgoing_edges:
            (c,d) = edge.opposite(grid[a][b])
            (edgeval1, edgeval2) = edge.value
            if edgeval1==0:
                continue
            e1 = edgeval1 + expu
            e2 = edgeval2 + expv

            if e1>0 and e2 >0:
                continue


            # Otherwise ((c,d),(e1,e2)) should be in grading_bases(i-1,j-1)
            target_index = grading_bases[(i-1,j-1)].index(((c,d),(e1,e2)))
            M_entries[source_index][target_index] += 1
            
    M = Matrix(R, M_entries)
    return M




def find_boundary_map_v(grid, grading_bases, grading):
    (i,j) = grading
    if (i,j) not in grading_bases:
        grading_bases[(i,j)] = find_grading_basis(grid, (i,j))
    if (i-1,j-1) not in grading_bases:
        grading_bases[(i-1, j-1)] = find_grading_basis(grid, (i-1,j-1))

    n = len(grading_bases[(i,j)])
    m = len(grading_bases[(i-1,j-1)])
    R = GF(2)
    M_entries = [[0 for x in range(m)] for y in range(n)]

    for source_index in range(n):
        ((a,b), (expu, expv)) = grading_bases[(i,j)][source_index]

        for edge in grid[a][b].outgoing_edges:
            (c,d) = edge.opposite(grid[a][b])
            (edgeval1, edgeval2) = edge.value

            if edgeval2==0:
                continue
            e1 = edgeval1 + expu
            e2 = edgeval2 + expv

            if e1>0 and e2 >0:
                continue


            # Otherwise ((c,d),(e1,e2)) should be in grading_bases(i-1,j-1)
            target_index = grading_bases[(i-1,j-1)].index(((c,d),(e1,e2)))
            M_entries[source_index][target_index] += 1
            
    M = Matrix(R, M_entries)
    return M



def mapu(grid, grading_bases, grading, vect, k):
    """
    Given a vector, vect, in a grading (gr1,gr2), in basis grading_bases[(gr1,gr2)], 
    find u^k*vect in basis grading_bases[(gr1-2*k,gr2)]
    Returns a vector over GF(2)
    """

    (gr1,gr2) = grading
    if (gr1,gr2) not in grading_bases:
        grading_bases[(gr1,gr2)]= find_grading_basis(grid, (gr1,gr2))
    if (gr1-2*k, gr2) not in grading_bases:
        grading_bases[(gr1-2*k,gr2)]= find_grading_basis(grid, (gr1-2*k,gr2))

    image = vector(GF(2), [0 for foo in range(len(grading_bases[(gr1-2*k,gr2)]))])
    for i in range(len(grading_bases[(gr1,gr2)])):
        if vect[i] != 0:
            ((spos1,spos2),(expu,expv)) = grading_bases[(gr1,gr2)][i]
            # above is source basis vector
            if expv>0 and expu + k>0:
                continue
            target_basis = ((spos1, spos2), (expu+k,expv))
            target_index = grading_bases[(gr1-2*k, gr2)].index(target_basis)
            image[target_index] += 1

    return image




def mapv(grid, grading_bases, grading, vect, k):
    """
    Given a vector, vect, in a grading (gr1,gr2), in basis grading_bases[(gr1,gr2)], 
    find u^k*vect in basis grading_bases[(gr1-2*k,gr2)]
    Returns a vector over GF(2)
    """

    (gr1,gr2) = grading
    if (gr1,gr2) not in grading_bases:
        grading_bases[(gr1,gr2)]= find_grading_basis(grid, (gr1,gr2))
    if (gr1, gr2-2*k) not in grading_bases:
        grading_bases[(gr1,gr2-2*k)]= find_grading_basis(grid, (gr1,gr2-2*k))

    image = vector(GF(2), [0 for foo in range(len(grading_bases[(gr1,gr2-2*k)]))])
    for i in range(len(grading_bases[(gr1,gr2)])):
        if vect[i] != 0:
            ((spos1,spos2),(expu,expv)) = grading_bases[(gr1,gr2)][i]
            # above is source basis vector
            if expv+k>0 and expu>0:
                continue
            target_basis = ((spos1, spos2), (expu,expv+k))
            target_index = grading_bases[(gr1, gr2-2*k)].index(target_basis)
            image[target_index] += 1

    return image
                                              
def mapv_matrix(grid, grading_bases, grading, k):
    """
    Returns the matrix over GF(2), from grading = (gr1, gr2) to grading (gr1-2*k, gr2)
    """
    (gr1, gr2) = grading

    if (gr1,gr2) not in grading_bases:
        grading_bases[(gr1,gr2)]= find_grading_basis(grid, (gr1,gr2))
    if (gr1, gr2-2*k) not in grading_bases:
        grading_bases[(gr1,gr2-2*k)]= find_grading_basis(grid, (gr1,gr2-2*k))

    n = len(grading_bases[(gr1, gr2)])
    m = len(grading_bases[(gr1, gr2-2*k)])
    R = GF(2)
    M_entries = [[0 for x in range(m)] for y in range(n)]

    for source_index in range(n):
        ((spos1,spos2), (expu, expv)) = grading_bases[(gr1,gr2)][source_index]
        if expu >0 and expv+k>0:
            continue

        target_basis = ((spos1, spos2), (expu,expv+k))
        target_index = grading_bases[(gr1, gr2-2*k)].index(target_basis)
        M_entries[source_index][target_index] += 1
        
    return Matrix(R, M_entries)


def mapu_matrix(grid, grading_bases, grading, k):
    """
    Returns the matrix over GF(2), from grading = (gr1, gr2) to grading (gr1-2*k, gr2)
    """
    (gr1, gr2) = grading

    if (gr1,gr2) not in grading_bases:
        grading_bases[(gr1,gr2)]= find_grading_basis(grid, (gr1,gr2))
    if (gr1-2*k, gr2) not in grading_bases:
        grading_bases[(gr1-2*k,gr2)]= find_grading_basis(grid, (gr1-2*k,gr2))

    n = len(grading_bases[(gr1, gr2)])
    m = len(grading_bases[(gr1-2*k, gr2)])
    R = GF(2)
    M_entries = [[0 for x in range(m)] for y in range(n)]

    for source_index in range(n):
        ((spos1,spos2), (expu, expv)) = grading_bases[(gr1,gr2)][source_index]
        if expu + k>0 and expv>0:
            continue

        target_basis = ((spos1, spos2), (expu+k,expv))
        target_index = grading_bases[(gr1-2*k, gr2)].index(target_basis)
        M_entries[source_index][target_index] += 1
        
    return Matrix(R, M_entries)









def addu(seq, grid, grading, grading_bases, potentials, u_boundary_maps, v_boundary_maps):
    """
    Each potential is a vector in grading_bases(grading)
    """

    potentials = map(lambda x:tuple(x), potentials)
    potentials = list(dict.fromkeys(potentials))
    potentials = map(lambda x:vector(GF(2), x), potentials)
    # This is to remove duplicates. It's not necessary, but maybe makes it go faster?
    
    print "in addu with sequence: " + str(seq)

    infty = 1000000
    maxu =  -infty

    if grading not in u_boundary_maps:
        u_boundary_maps[grading] = find_boundary_map_u(grid, grading_bases, grading)
        
    du = u_boundary_maps[grading]

    # maxu will store the largest number of factors of u that can go into
    # du(potential) for a potential

    (gr1, gr2) = grading

    best_potentials = []

    for potential in potentials:
        image = potential * du
        if image == vector(GF(2), [0 for foo in range(du.ncols())]):
            if maxu<infty:
                maxu = infty
                best_potentials = []

            if maxu == infty: #Note this will happen if previous if statement happens
                best_potentials.append(potential)
            continue
        
        # Otherwise, potential*du is non-zero, so we are interested in
        # how many factors of u it contains
        
        minu_in_image = infty
        
        for i in range(len(image)):
            if image[i] == 1:
                ((p1, p2), (expu, expv)) = grading_bases[(gr1-1,gr2-1)][i]
                if expu<minu_in_image:
                    minu_in_image = expu
        if minu_in_image>maxu:
            best_potentials = []
            maxu = minu_in_image

        if minu_in_image == maxu: #Note this will happen if previous if statement happens
            best_potentials.append(potential)
            

        
    if maxu == infty:
        new_best = []
        # at this point we want to see how many we can map in.
        # see if we can map in u^k*potential for k starting at 1
        found_one = False
        maxk = 0
        next_potentials = []
    
        for k in range(1, 50): # cont down 50, 49, ... 1.
            prev_grading = (gr1-2*k+1, gr2+1)
            # this is the grading we want to be mapping in from
            
            if prev_grading not in u_boundary_maps:
                u_boundary_maps[prev_grading] = find_boundary_map_u(
                    grid, grading_bases, prev_grading)
            du_prev = u_boundary_maps[prev_grading]
            phi = linear_transformation(du_prev)
            
            if len(grading_bases[prev_grading]) == 0:
                #This means we have reached the end, probably
                continue
            

            for potential in best_potentials:
                try:
                    preim = phi.lift(mapu(grid, grading_bases, grading, potential, k))
                    if not found_one:
                        maxk = k
                        found_one = True
                    ker = kernel(du_prev)
                    for ker_vect in ker:
                        next_potentials.append(preim + ker_vect)
                    
                    
                except ValueError:
                    # So this is when u^k (potential) is not in image
                    continue
            if found_one:
                new_seq = copy.deepcopy(seq)
                new_seq.append(maxk)
                return addv(new_seq, grid, prev_grading, grading_bases, next_potentials, u_boundary_maps, v_boundary_maps)

        if not found_one:
            # We weren't able to build more, so just end here
            return copy.deepcopy(seq)
       
    else:
        new_seq = copy.deepcopy(seq)
        next_grading = (gr1-1+2*maxu, gr2-1)
        new_seq.append(-maxu)
        umat = mapu_matrix(grid, grading_bases, next_grading, maxu)
        phi = linear_transformation(umat)
        next_potentials = []
        #next_potentials is where each best_potential maps to
        # so it's x such that u^maxu x= du potential
        # but wait, we want to know the kernel of u^maxu for this.
        # So we have to re-write mapu to make it return the matrix. 
                       

        for potential in best_potentials:
            first_next = phi.lift(potential * du)
            ker = kernel(umat)
            for ker_vect in ker:
                next_potentials.append(first_next + ker_vect)

        return addv(new_seq, grid, next_grading, grading_bases, next_potentials, u_boundary_maps, v_boundary_maps)
        
                                       







def addv(seq, grid, grading, grading_bases, potentials, u_boundary_maps, v_boundary_maps):
    """
    Each potential is a vector in grading_bases(grading)
    """
    
    print "in addv with sequence: " + str(seq)

    potentials = map(lambda x:tuple(x), potentials)
    potentials = list(dict.fromkeys(potentials))
    potentials = map(lambda x:vector(GF(2), x), potentials)
    # This is to remove duplicates. It's not necessary, but maybe makes it go faster?
    


    infty = 1000000
    maxv =  -infty

    if grading not in v_boundary_maps:
        v_boundary_maps[grading] = find_boundary_map_v(grid, grading_bases, grading)
        


    dv = v_boundary_maps[grading]

    # maxv will store the largest number of factors of v that can go into
    # dv(potential) for a potential

    (gr1, gr2) = grading

    best_potentials = []

    for potential in potentials:
        image = potential * dv
        if image == vector(GF(2), [0 for foo in range(dv.ncols())]):
            if maxv<infty:
                maxv = infty
                best_potentials = []

            if maxv == infty: #Note this will happen if previous if statement happens
                best_potentials.append(potential)
            continue

        # Otherwise, potential*dv is non-zero, so we are interested in
        # how many factors of v it contains
        
        minv_in_image = infty
        
        for i in range(len(image)):
            if image[i] == 1:
                ((p1, p2), (expu, expv)) = grading_bases[(gr1-1,gr2-1)][i]
                if expv<minv_in_image:
                    minv_in_image = expv
        if minv_in_image>maxv:
            best_potentials = []
            maxv = minv_in_image

        if minv_in_image == maxv: #Note this will happen if previous if statement happens
            best_potentials.append(potential)
            

        
    if maxv == infty:
        new_best = []
        # at this point we want to see how many we can map in.
        # see if we can map in u^k*potential for k starting at 1
        found_one = False
        maxk = 0
        next_potentials = []
    
        for k in range(1, 50): # cont down 50, 49, ... 1.
            prev_grading = (gr1+1, gr2-2*k+1)

            for potential in best_potentials:

                if prev_grading not in v_boundary_maps:
                    v_boundary_maps[prev_grading] = find_boundary_map_v(
                        grid, grading_bases, prev_grading)
                dv_prev = v_boundary_maps[prev_grading]
                phi = linear_transformation(dv_prev)

                if len(grading_bases[prev_grading]) == 0:
                    #This means we have reached the end, probably
                    continue
            

                try:
                    preim = phi.lift(mapv(grid, grading_bases, grading, potential, k))
                    if not found_one:
                        maxk = k
                        found_one = True
                    ker = kernel(dv_prev)
                    for ker_vect in ker:
                        next_potentials.append(preim + ker_vect)
                    
                    
                except ValueError:
                    # So this is when u^k (potential) is not in image
                    continue
            if found_one:
                new_seq = copy.deepcopy(seq)
                new_seq.append(maxk)
                return addu(new_seq, grid, prev_grading, grading_bases, next_potentials, u_boundary_maps, v_boundary_maps)

        if not found_one:
            # We weren't able to build more, so just end here
            return copy.deepcopy(seq)
       
    else:
        new_seq = copy.deepcopy(seq)
        next_grading = (gr1-1, gr2-1+2*maxv)
        new_seq.append(-maxv)
        vmat = mapv_matrix(grid, grading_bases, next_grading, maxv)

        phi = linear_transformation(vmat)
        next_potentials = []
        #next_potentials is where each best_potential maps to
        # so it's x such that u^maxu x= du potential
        # but wait, we want to know the kernel of u^maxu for this.
        # So we have to re-write mapu to make it return the matrix. 
                       

        for potential in best_potentials:

            first_next = phi.lift(potential * dv)
            
            ker = kernel(vmat)
            for ker_vect in ker:
                next_potentials.append(first_next + ker_vect)

        return addu(new_seq, grid, next_grading, grading_bases, next_potentials, u_boundary_maps, v_boundary_maps)
        
                                       








        
    
if __name__ == '__main__':
    print "GOT HERE"

    
    f = open("input.txt", 'r')
    output = open("results.txt", 'w')
    num_egs = int(f.readline())
    for num in range(num_egs):
        line = f.readline()
        chain1_seq = line.split(',')
        chain1_seq = map(int,chain1_seq)
        chain1 = make_chain(chain1_seq)
        line = f.readline()
        chain2_seq = line.split(',')
        chain2_seq = map(int,chain2_seq)
        chain2 = make_chain(chain2_seq)

        #chain2 = make_chain([1, -2, 2, -1])
        grid = make_grid(chain1, chain2)

        grading_bases = {}
        u_boundary_maps = {}     
        v_boundary_maps = {}
        # u_boundary_maps[(0,0)] stores the u boundary map in grading (0,0)
        # as a maptrix with basis grading_bases[(0,0)]

        u_boundary_maps[(0,0)] = find_boundary_map_u(grid, grading_bases, (0,0))
        v_boundary_maps[(0,0)] = find_boundary_map_v(grid, grading_bases, (0,0))

        grading_bases[(0,0)] = find_grading_basis(grid, (0,0))
        corner_basis_index = -1
        # This is an int that stores the index of ((0,0),(0,0))
        # in the grading basis of the (0,0) grading.
    
        for i in range(len(grading_bases[(0,0)])):
            if grading_bases[(0,0)][i] == ((0,0),(0,0)):
                corner_basis_index = i
                break


        potentials = []

        n = len(grading_bases[(0,0)])-1
        for N in range(2**n):
            over_under_string = "{0:b}".format(N)
            while len(over_under_string)<n:
                over_under_string = "0" + over_under_string

            potential = map(int, over_under_string)

            potential = potential[:corner_basis_index] + [1,] + potential[corner_basis_index:]
        
            if n == 0:
                potential = [1,]
            potential = vector(GF(2), potential)
            # Note that in sage, vectors are horizontal,
            # so multiplication of matrix by vector
            # is on the right.
            dv = v_boundary_maps[(0,0)]
            print potential
            print dv
            if potential * dv == vector(GF(2), [0 for foo in range(dv.ncols())]):

                potentials.append(potential)

        seq = addu([], grid, (0,0), grading_bases, potentials, u_boundary_maps, v_boundary_maps)

        print "Answer is: " + str(seq)
        output.write(str(seq) + "\n")
        output.flush()

        
