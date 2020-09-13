class EmptyCell:
    def __init__(self, id, x):
        self.id = id
        self.x = x

    def vertices(self):
        return None

    def face_vertices(self):
        return None

    def neighbors(self):
        return None