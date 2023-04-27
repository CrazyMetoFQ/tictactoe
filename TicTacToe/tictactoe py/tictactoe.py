from numpy import array


def is_part_of_2(x,y):
    """
    checks if x:iterable is part of y:iterable 
    SCOPE 2
    """
#    print(x)
    l = [[j in i for j in x] for i in y]
    for i in l:
        if False not in i:
            return True
    return False


class Board:
    
    @staticmethod
    def generate_board(n:int,place_holder='-'):
        # fr sm rsn [["-"]*n]*n wont work properly
        # prbly could have just used array.reshape()
        
        ls = [[place_holder for i in range(n)] for i in range(n)]
        return ls
    
        
    def __init__(self,n:int,w:int=3):
        
        self.n = n
        self.board = Board.generate_board(n)
        self.w = w
    
    def show(self):
        
        for i in self.board:
            for n,j in enumerate(i):
                if n!= len(i)-1:
                    print(f" {j} |", end='')
                else:
                    print(f" {j}", end='\n')
    
    def change(self, ch="-",pos=[0,0], brd=None):
        """
        ch = X|O
        pos-> coord
        """
        
        if brd==None:
            self.board[pos[1]][pos[0]] = ch
        else:
            n = int(len(brd)**0.5)
            self.board = array(brd).reshape(n,n).tolist()
    
    def wincheck(self, ch):
        
        # scalar done on array
        # converted to list for comparistion

        
        # get all coord of ch
        coords = array([array((x,y)) for y,row in enumerate(self.board) for x,i in enumerate(row) if i==ch])
        
        # ltr Diagnal check
        dg_ls = [True for n in range(len(coords)) if [-1,-1] in [list(i) for i in coords[n]-coords]]
        if self.w - 1 <= len(dg_ls):
            print("Diagnal ltr")
            return True
            
        # rtl Diagnal check
        dg_ls = [True for n in range(len(coords)) if [1,-1] in [list(i) for i in coords[n]-coords]]
        if self.w - 1 <= len(dg_ls):
            print("Diagnal rtl")
            return True

        
        ls = array([coords[n]-coords for n in range(len(coords))])
        ls = ls.tolist()
        
        # Vertical check
        if is_part_of_2([[0,i] for i in range(self.w)],ls) or is_part_of_2([[0,-i] for i in range(self.w)],ls):
            print("Vertical")
            return True

        # Horizontal check
        h_ls = [([list(i) for i in coords[n]-coords]) for n in range(len(coords))]
        if is_part_of_2([[i,0] for i in range(self.w)], ls) or is_part_of_2([[-i,0] for i in range(self.w)], ls):
            print("Horizontal")
            return True
        
        # return coords, ls
        return False
        


# brd = Board(5,4)



# brd.change('X',(1,4))
# brd.change('X',(2,3))
# brd.change('X',(3,2))
# brd.change('X',(4,1))


# brd.show()

# coords, ls = brd.wincheck("X")

