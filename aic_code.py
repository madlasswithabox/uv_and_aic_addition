from sage.all import *
import copy


"""
This is like aic_take1, except that instead of just returning the sequence,
we return the sequence and a list of possible places each could map to
"""


class Vertex:
    
    """
    Signifies a copy of R.

    attributes:
    * grading. This is a single integer, not a tuple
    * Since this is the AIC version, there is no actual u grading!
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
        self.ld = None
        self.lu = None
        self.rd = None
        self.ru = None

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

        if self.ld != None:
            if self.ld.direction == "ld":
                self.outgoing_edges.append(self.ld)
            else:
                self.incoming_edges.append(self.ld)
        
        if self.lu != None:
            if self.lu.direction == "lu":
                self.outgoing_edges.append(self.lu)

            else:
                self.incoming_edges.append(self.lu)


        if self.rd != None:
            if self.rd.direction == "rd":
                self.outgoing_edges.append(self.rd)
            else:
                self.incoming_edges.append(self.rd)
        
        if self.ru != None:
            if self.ru.direction == "ru":
                self.outgoing_edges.append(self.ru)

            else:
                self.incoming_edges.append(self.ru)



                

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


    
def make_gradings_chain(chain):
    """
    seq is an array of vertices, which do not have gradings set,
    we will set the gradings.
    Now the omega maps (the ones from 2i to 2i+1, so t is odd, 
    just preserve grading
    """
    n = len(chain)

    chain[0].grading = 0
    for t in range(1,n):
        v = chain[t]
        prev1 = chain[t-1].grading
        e = v.left
        if t % 2 == 0:
            (i,j) = e.value
            if e.direction == "left":
                im1 = prev1 -2*j
                # These are the gradings of the image of 1, (this image is in previous)
                v.grading = im1+1

            elif e.direction == "right":
                im1 = prev1 -1
                # These are the gradings of the image of 1 (this image is in current)
                v.grading = im1+2*j

        else:
            v.grading = prev1


    


def make_gradings_grid(grid):
    """
    Making the gradings, assuming that the gradings on the 0'th row is already correct, 
    which it should be if you have down 
    """
    for t in range(1,len(grid)): # t records the row, so t = 0 would be the bottom row
        for t1 in range(len(grid[t])):
            v = grid[t][t1]
            prev1 = grid[t-1][t1].grading # this the one below

            if t % 2 == 0:
                e = v.down
            
                (i,j) = e.value
                if e.direction == "down":
                    im1 = prev1 -2*j
                    # These are the gradings of the image of 1, (this image is in previous)
                    v.grading = im1+1

                elif e.direction == "up":
                    im1 = prev1 -1
                    # These are the gradings of the image of 1 (this image is in current)
                    v.grading = im1+2*j
            else:
                v.grading = prev1


    
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
    """
    TO DO: Modify this to account for the diagonal maps
    """
    grid = [] # each item is a copy of chain1, up is going up in coordinate
    # So grid[t+1] is above grid[t]
    for t in range(len(chain2)):
        grid.append(copy.deepcopy(chain1))
        for t1 in range(len(grid[t])):
            grid[t][t1].position = (t,t1)

    # ADD VERTICAL MAPS:
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

    #ADD DIAGONAL MAPS:
    for t in range(len(chain2)/2):
        for t1 in range(len(chain1)/2):
            #diagonal map in box (2*t, 2*t1), (2*t+1, 2*t1 + 1), (2*t, 2*t1+1), (2*t + 1, 2*t1)
            edge = Edge()
            edge.value = (1,0)

            if(chain1[2*t1].right.direction == "right" and chain2[2*t].right.direction == "right"):
                edge.direction = "ru"
            elif(chain1[2*t1].right.direction == "right" and chain2[2*t].right.direction == "left"):
                edge.direction = "rd"
            elif(chain1[2*t1].right.direction == "left" and chain2[2*t].right.direction == "right"):
                edge.direction = "lu"
            elif(chain1[2*t1].right.direction == "left" and chain2[2*t].right.direction == "left"):
                edge.direction = "ld"

            if(edge.direction == "rd" or edge.direction == "lu"):
                edge.v1 = grid[2*t][2*t1+1]
                grid[2*t][2*t1+1].lu = edge
                edge.v2 = grid[2*t+1][2*t1]
                grid[2*t+1][2*t1].rd = edge

            else:
                edge.v1 = grid[2*t][2*t1]
                grid[2*t][2*t1].ru = edge
                edge.v2 = grid[2*t+1][2*t1+1]
                grid[2*t+1][2*t1+1].ld = edge
          
            
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
            edges = [vertex.up, vertex.down, vertex.left, vertex.right,
                     vertex.lu, vertex.ld, vertex.ru, vertex.rd]
            print vertex.incoming_edges
            print vertex.outgoing_edges
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
    j = grading
    for row in grid:
        for v in row:
            j1 = v.grading
            if j1 >= j and (j1-j) % 2 == 0:
                basis.append( (v.position, (0,(j1-j)/2) ) )
                
    return basis



def find_boundary_map_u(grid, grading_bases, grading):
    """
    Now, this is what the u maps do, so at this point
    they should not affect the grading
    So now it's a square matrix, seeing as it goes from a 
    grading to itself
    """
    j = grading
    if j not in grading_bases:
        grading_bases[j] = find_grading_basis(grid, j)

    n = len(grading_bases[j])
    #R = IntegerModRing(2)
    R = GF(2)
    M_entries = [[0 for x in range(n)] for y in range(n)]
    for source_index in range(n):
        ((a,b), (expu, expv)) = grading_bases[j][source_index]
        for edge in grid[a][b].outgoing_edges:
            (c,d) = edge.opposite(grid[a][b])
            (edgeval1, edgeval2) = edge.value
            if edgeval2>0: # This means it's actually a v map
                continue

            # ((c,d),(expu,expv)) is what we map to
            target_index = grading_bases[j].index(((c,d),(expu,expv)))

            
            M_entries[source_index][target_index] += 1
            
    M = Matrix(R, M_entries)
    return M




def find_boundary_map_v(grid, grading_bases, grading):
    j = grading
    if j not in grading_bases:
        grading_bases[j] = find_grading_basis(grid, j)
    if j-1 not in grading_bases:
        grading_bases[j-1] = find_grading_basis(grid, j-1)

    n = len(grading_bases[j])
    m = len(grading_bases[j-1])
    R = GF(2)
    M_entries = [[0 for x in range(m)] for y in range(n)]

    for source_index in range(n):
        ((a,b), (expu, expv)) = grading_bases[j][source_index]

        for edge in grid[a][b].outgoing_edges:
            (c,d) = edge.opposite(grid[a][b])
            (edgeval1, edgeval2) = edge.value

            if edgeval2==0:
                continue
            e1 = edgeval1 + expu
            e2 = edgeval2 + expv

            if e1>0 and e2 >0:
                continue


            # Otherwise ((c,d),(e1,e2)) should be in grading_bases[j-1]
            target_index = grading_bases[j-1].index(((c,d),(e1,e2)))
            M_entries[source_index][target_index] += 1
            
    M = Matrix(R, M_entries)
    return M



def mapu(grid, grading_bases, grading, vect, k):
    """
    Given a vector, vect, in a grading gr2, in basis grading_bases[gr2], 
    find u^k*vect in basis grading_bases[gr2]
    Returns a vector over GF(2)
    """

    return copy.deepcopy(vect)




def mapv(grid, grading_bases, grading, vect, k):
    """
    Given a vector, vect, in a grading gr2, in basis grading_bases[gr2], 
    find v^k*vect in basis grading_bases[gr2-2*k]
    Returns a vector over GF(2)
    """

    gr2 = grading
    if gr2 not in grading_bases:
        grading_bases[gr2]= find_grading_basis(grid, gr2)
    if gr2-2*k not in grading_bases:
        grading_bases[gr2-2*k]= find_grading_basis(grid, gr2-2*k)

    image = vector(GF(2), [0 for foo in range(len(grading_bases[gr2-2*k]))])
    for i in range(len(grading_bases[gr2])):
        if vect[i] != 0:
            ((spos1,spos2),(expu,expv)) = grading_bases[gr2][i]
            # above is source basis vector
            if expv+k>0 and expu>0:
                continue
            target_basis = ((spos1, spos2), (expu,expv+k))
            target_index = grading_bases[gr2-2*k].index(target_basis)
            image[target_index] += 1

    return image
                                              
def mapv_matrix(grid, grading_bases, grading, k):
    """
    Returns the matrix over GF(2), from grading = gr2 to grading (gr2-2k)
    """
    gr2 = grading

    if gr2 not in grading_bases:
        grading_bases[gr2]= find_grading_basis(grid, gr2)
    if gr2-2*k not in grading_bases:
        grading_bases[gr2-2*k]= find_grading_basis(grid, gr2-2*k)

    n = len(grading_bases[gr2])
    m = len(grading_bases[gr2-2*k])
    R = GF(2)
    M_entries = [[0 for x in range(m)] for y in range(n)]

    for source_index in range(n):
        ((spos1,spos2), (expu, expv)) = grading_bases[gr2][source_index]
        if expu >0 and expv+k>0:
            continue

        target_basis = ((spos1, spos2), (expu,expv+k))
        target_index = grading_bases[gr2-2*k].index(target_basis)
        M_entries[source_index][target_index] += 1
        
    return Matrix(R, M_entries)


def mapu_matrix(grid, grading_bases, grading, k):
    """
    This is actually just the identity, because there is no u.
    """
    gr2 = grading

    if gr2 not in grading_bases:
        grading_bases[gr2]= find_grading_basis(grid,gr2)

    n = len(grading_bases[gr2])
    R = GF(2)
    M_entries = [[0 for x in range(n)] for y in range(n)]

    for source_index in range(n):
        M_entries[source_index][source_index]  = 1
        
    return Matrix(R, M_entries)









def addu(seq, grid, grading, grading_bases, potentials, u_boundary_maps, v_boundary_maps):
    """
    Each potential is a vector in grading_bases(grading)
    which is the grading of the last vertex we added
    """

    potentials = map(lambda x:tuple(x), potentials)
    potentials = list(dict.fromkeys(potentials))
    potentials = map(lambda x:vector(GF(2), x), potentials)
    # This is to remove duplicates. It's not necessary, but maybe makes it go faster?


    grading_dim = len(grading_bases[grading])
    for i in range(grading_dim):
        if grading_bases[grading][i][1][1] == 0:
            continue
        # So grading_bases[prev_grading][i] represents a (position, (0, expv))
        additional_potentials = []
        adding_vector = vector(GF(2), [0 for foo in range(grading_dim)])
        adding_vector[i] = 1
        for vect in potentials:
            additional_potentials.append(vect + adding_vector)
        potentials = potentials + additional_potentials
                    
        # Now removing duplicates:
        potentials = map(lambda x:tuple(x), potentials)
        potentials = list(dict.fromkeys(potentials))
        potentials = map(lambda x:vector(GF(2), x), potentials)


    print ""
    print "in addu with sequence: " + str(seq)
    print "with " + str(len(potentials)) + " distinct potentials"
    #print "grading_bases[grading]: "+ str(grading_bases[grading])
    #print "Potentials:" 
    #for potential in potentials:
    #    print potential


    infty = 1000000
    maxu =  -infty

    if grading not in u_boundary_maps:
        u_boundary_maps[grading] = find_boundary_map_u(grid, grading_bases, grading)
        
    du = u_boundary_maps[grading]

    # maxu will store the largest number of factors of u that can go into
    # du(potential) for a potential

    gr2 = grading

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
        
        # Otherwise, potential*du is non-zero, so we have to have a map to the right
        # If maxu winds up not infinity, then that means no potentials are places we
        # can end, so all potentials require further maps to the right.
        



    if maxu == infty:
        new_best = []
        # at this point we want to see whether we can map left
        found_one = False
        maxk = 0
        next_potentials = []
    
        for k in range(1,2):
            # just want to see whether we can map something into
            # the grid so that du of it goes to the last potential.
            prev_grading =  gr2 
            # if we can map left, previous guy (which is to the right) is in this grading
            
            if prev_grading not in u_boundary_maps:
                u_boundary_maps[prev_grading] = find_boundary_map_u(
                    grid, grading_bases, prev_grading)
            du_prev = u_boundary_maps[prev_grading]
            phi = linear_transformation(du_prev)
            if len(grading_bases[prev_grading]) == 0:
                # This should not happen, because potential is already in this grading,
                # but whatever
                print "Huge problem 1"
                return "Huge problem 1"
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
                
                # At this point about to call addv. the next_potentials are in prev_grading
                # And we need to add in other things in prev_grading that are multiples of v

                (returned_seq, returned_pot_list) = addv(new_seq, grid, prev_grading, grading_bases, next_potentials, u_boundary_maps, v_boundary_maps)
                return (returned_seq, [(potentials, grading)] + returned_pot_list)



        if not found_one:
            # We weren't able to build more, so just end here
            return (copy.deepcopy(seq), [(best_potentials, grading)])
       
    else:
        new_seq = copy.deepcopy(seq)
        next_grading = gr2
        new_seq.append(-1)
        umat = mapu_matrix(grid, grading_bases, next_grading, maxu)
        phi = linear_transformation(umat)
        next_potentials = []
        #next_potentials is where each best_potential maps to
        # so it's x such that u^maxu x= du potential
        # but wait, we want to know the kernel of u^maxu for this.
        # So we have to re-write mapu to make it return the matrix.
        # ALSO, ACTUALLY IN THIS CASE phi = Identity
                       
        for potential in potentials:
            first_next = phi.lift(potential * du)
            ker = kernel(umat)
            for ker_vect in ker:
                next_potentials.append(first_next + ker_vect)
                
        # At this point about to call addv. the next_potentials are in next_grading
        # And we need to add in other things in next_grading that are multiples of v
        grading_dim = len(grading_bases[next_grading])
        for i in range(grading_dim):
            if grading_bases[next_grading][i][1][1] == 0:
                continue

            # So grading_bases[next_grading][i] represents a (position, (0, expv))
            additional_potentials = []
            adding_vector = vector(GF(2), [0 for foo in range(grading_dim)])
            adding_vector[i] = 1
            for vect in next_potentials:
                additional_potentials.append(vect + adding_vector)
                
            next_potentials = next_potentials + additional_potentials

            #Now removing duplicates:
            next_potentials = map(lambda x:tuple(x), next_potentials)
            next_potentials = list(dict.fromkeys(next_potentials))
            next_potentials = map(lambda x:vector(GF(2), x), next_potentials)
         

        
                                       

        (returned_seq, returned_pot_list) = addv(new_seq, grid, next_grading, grading_bases, next_potentials, u_boundary_maps, v_boundary_maps)
        
        return (returned_seq, [(potentials, grading)] + returned_pot_list)





def addv(seq, grid, grading, grading_bases, potentials, u_boundary_maps, v_boundary_maps):
    """
    Each potential is a vector in grading_bases(grading)
    """

    print ""

    
    print "in addv with sequence: " + str(seq)


    potentials = map(lambda x:tuple(x), potentials)
    potentials = list(dict.fromkeys(potentials))
    potentials = map(lambda x:vector(GF(2), x), potentials)
    # This is to remove duplicates. It's not necessary, but maybe makes it go faster?
    

    print "with " + str(len(potentials)) + " distinct potentials"


    #print "grading_bases[grading]: "+ str(grading_bases[grading])
    #print "Potentials:" 
    #for potential in potentials:
    #    print potential



    infty = 1000000
    maxv =  -infty

    if grading not in v_boundary_maps:
        v_boundary_maps[grading] = find_boundary_map_v(grid, grading_bases, grading)
        


    dv = v_boundary_maps[grading]

    # maxv will store the largest number of factors of v that can go into
    # dv(potential) for a potential

    gr2 = grading

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
                ((p1, p2), (expu, expv)) = grading_bases[gr2-1][i]
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
            prev_grading = gr2-2*k+1

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
                (returned_seq, returned_pot_list) =  addu(new_seq, grid, prev_grading, grading_bases, next_potentials, u_boundary_maps, v_boundary_maps)

                return (returned_seq, [(potentials, grading)] + returned_pot_list)



        if not found_one:
            # We weren't able to build more, so just end here
            return (copy.deepcopy(seq), [(best_potentials,grading)])
       
    else:
        new_seq = copy.deepcopy(seq)
        next_grading = gr2-1+2*maxv
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


        (returned_seq, returned_pot_list)  =  addu(new_seq, grid, next_grading, grading_bases, next_potentials, u_boundary_maps, v_boundary_maps)
        
                                       
        return (returned_seq, [(potentials, grading)] + returned_pot_list)

                                       








        
    
if __name__ == '__main__':
    print "GOT HERE"

    
    f = open("aic_input.txt", 'r')
    #output = open("aic_results.txt", 'w')
    seq_output = open("aic_results.txt", 'w')

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

        grid = make_grid(chain1, chain2)


        grading_bases = {}
        u_boundary_maps = {}     
        v_boundary_maps = {}
        # u_boundary_maps[0] stores the u boundary map in grading 0
        # as a maptrix with basis grading_bases[0]

        u_boundary_maps[0] = find_boundary_map_u(grid, grading_bases, 0)
        v_boundary_maps[0] = find_boundary_map_v(grid, grading_bases, 0)

        grading_bases[0] = find_grading_basis(grid, 0)
        corner_basis_index = -1
        # This is an int that stores the index of ((0,0),(0,0))

        # in the grading basis of the (0,0) grading.
    
        for i in range(len(grading_bases[0])):
            if grading_bases[0][i] == ((0,0),(0,0)):
                corner_basis_index = i
                break
        #Above, we find corner_basis_index, because starting positions must have 1 there.

        potentials = []


        n = len(grading_bases[0])-1
        print n
        #if n>14:
        #    print "TL;DR"
        #    output.write("TL;DR \n")
        #    output.flush()
        #    continue
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
            dv = v_boundary_maps[0]
            #print potential
            #print dv
            if potential * dv == vector(GF(2), [0 for foo in range(dv.ncols())]):

                potentials.append(potential)
#        print grading_bases[0]
#        for potential in potentials:
#            print potential
            
        
        (seq,end_potentials_list) = addu([], grid, 0, grading_bases, potentials, u_boundary_maps, v_boundary_maps)
        print "Answer is: " + str(seq)
        #output.write(str(seq) + "\n")


        #seq_output.write("Sequence1: " + str(chain1_seq) + "\n")
        #seq_output.write("Sequence2: " + str(chain2_seq) + "\n")
        seq_output.write(str(seq) + "\n")




        #for i in range(len(end_potentials_list)):
        #    output.write("1 in R_" + str(i) + " could map to potentials:\n")
        #    (end_potentials, grading)= end_potentials_list[i]
        #    for potential in end_potentials:
        #        output.write(str(potential) + "\n")

        #    output.write("In basis:\n")
        #    output.write(str(grading_bases[grading]) + "\n")
        #    output.write("\n")
            
        #output.flush()
        seq_output.flush()
