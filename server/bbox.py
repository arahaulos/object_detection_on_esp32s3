
def axis_overlap(a0, a1, b0, b1):
    #check if a is inside b
    if (a0 >= b0 and a1 <= b1):
        return a1 - a0
    
    #check if b is inside a
    if (b0 >= a0 and b1 <= a1):
        return b1 - b0
    
    if (a1 <= b0 or a0 >= b1):
        return 0.0
    elif (a0 >= b0):
        return b1 - a0
    elif (b0 >= a0):
        return a1 - b0
    else:
        return 0.0
    
class bbox:
    def __init__(self, ot, co, et, x, y, w, h):
        self.object_type = ot
        self.confidence = co
        self.estimated_distance = et
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def scaled(self, sx, sy):
        return bbox(self.object_type, self.confidence, self.estimated_distance, self.x * sx, self.y * sy, self.w * sx, self.h * sy)

    def dumb(self):
        print("Pos: {}x{}  Size: {}x{}  Confidence {}".format(self.x, self.y, self.w, self.h, self.confidence))

    def area(self):
        return self.w*self.h
    
    def area_of_overlap(self, other):
        return axis_overlap(self.x, self.x + self.w, other.x, other.x+other.w)*axis_overlap(self.y, self.y + self.h, other.y, other.y+other.h)
    
    def intersection_of_union(self, other):
        overlap = self.area_of_overlap(other)
        union = self.area() + other.area() - overlap

        #print("{} {}".format(overlap, union))

        return overlap / union