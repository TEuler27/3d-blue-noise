import numpy as np
from tess import Container
import scipy.spatial
import scipy.optimize
import scipy.sparse
import quadpy
import time
import multiprocessing
from functools import partial
import pyny3d.geoms as pyny
import pymesh
import pickle
from rejectionsampling import randomvariate
from EmptyCell import EmptyCell

class Optimizer:
    """
        The Optimizer class wraps all the routine and calculation
        needed to solve the blue noise problem in 3D

        Parameters:

        n: number of atoms
        a: the problem will be solved in the [-a,a]^3 cube
        rho: density function. rho should accept data of shape (3, k, m)
        and return data of the form (k, m)

        Arguments:

        X: actual points coordinates
        w: actual power diagram weight
        c: actual power diagram (object of type tess.Container)
        vertices: list of vertices
        facets: list of the faces of each cell
        neighbors: list of the neighbors of each cell

        Useful functions:

        solve: solve the problem
        saveNetwork: save the .obj file containing the final diagram
        quit: shuts down the pool object used for parallel integration
    """

    def __init__(self, n, a, rho):
        self.counter = 0
        self.rhoh = rho
        self.a = a
        self.n = n
        self.alphabound = 50
        self.maxalpha = 200
        self.isBoundIncreasable = False
        self.counterBound = 0
        self.X = self.randomvariate(rho).reshape(n,3)
        self.w = np.ones(self.n)
        self.pool = multiprocessing.Pool()
        self.totM = self.totalMass()
        self.m = self.totM / self.n
        self.c = self.voronoi()
        self.vertices = list(map(lambda x: np.array(x.vertices()), self.c))
        self.facets = list(map(lambda x: np.array(x.face_vertices()), self.c))
        self.neighbors = list(map(lambda x: np.array(x.neighbors()), self.c))
        self.masses = self.cellsIntegral(self.rho)
        self.grad = np.empty(4*self.n)
        self.grad[3 * self.n:] = self.m - self.masses
        self.grad[:3 * self.n] = (2 * self.masses.reshape(self.n, 1) * (self.X - self.baricentres())).reshape(self.n * 3)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if (not isinstance(v, Container)) and (not isinstance(v, multiprocessing.pool.Pool))}

    def rho(self,x,i):
        return self.rhoh(x) / self.totM

    def rhoface(self, x, param, i):
        missingCoord = (- param[3] - (x.T @ np.concatenate([param[:i], param[i+1:3]])).T) / param[i]
        xTrue = np.insert(x, i, missingCoord, axis = 0)
        da = np.abs(1 / param[i])
        return self.rho(xTrue, 0) * da

    def xrho(self,x,i):
        return x * self.rho(x,i)

    def crho(self,x,i):
        return self.rho(x,i) * (np.linalg.norm(x - self.X[i,:].reshape(3,1,1), axis = 0) ** 2)

    def crhoOutOfState(self,x,i,points):
        return self.rho(x,i) * (np.linalg.norm(x - points[i,:].reshape(3,1,1), axis = 0) ** 2)

    def randomvariate(self, pdf):
        x = np.mgrid[-self.a:self.a:50j, -self.a:self.a:50j, -self.a:self.a:50j]
        x = x.reshape(3, 1, 125000)
        y = pdf(x)
        pmin = 0.
        pmax = y.max()
        naccept = 0
        ran = []
        while naccept < self.n:
            x = np.random.uniform(-self.a, self.a, size=(3,1))
            y = np.random.uniform(pmin, pmax)
            if y < pdf(x):
                ran.append(x.T)
                naccept = naccept + 1
        ran = np.vstack(ran)
        return ran

    def cost(self):
        costPerCell = self.cellsIntegral(self.crho)
        return costPerCell.sum()

    def F(self):
        costPerCell = self.cellsIntegral(self.crho)
        return (costPerCell.sum() - self.w.reshape(1, self.n) @ (self.masses - self.m))[0]

    def FOutOfSTate(self, X, w):
        X = X.reshape(self.n, 3)
        c = self.voronoiOutOfState(X, w)
        masses = self.cellsIntegral(self.rho, c)
        costPerCell = self.cellsIntegral(partial(self.crhoOutOfState, points = X), c)
        return (costPerCell.sum() - w.reshape(1, self.n) @ (masses - self.m))[0]

    def totalMass(self):
        points = [[-self.a,-self.a,-self.a],[-self.a,-self.a,self.a],[-self.a,self.a,-self.a],[-self.a,self.a,self.a],[self.a,-self.a,-self.a],[self.a,-self.a,self.a],[self.a,self.a,-self.a],[self.a,self.a,self.a]]
        facets = [[0, 1, 3, 2],[2, 3, 7, 6],[4, 5, 7, 6],[0, 1, 5, 4], [0, 2, 6, 4], [1, 3, 7, 5]]
        baseTetra = scipy.spatial.Delaunay(points).simplices
        triangles = []
        for facet in facets:
            triangleMesh = pymesh.triangle()
            triangleMesh.points = np.array(points)[facet]
            triangleMesh.verbosity = 0
            triangleMesh.max_num_steiner_points = 0
            triangleMesh.run()
            triangles.append(np.array(facet)[triangleMesh.mesh.faces])
        triangles = np.vstack(triangles)
        tetgen = pymesh.tetgen()
        tetgen.points = points
        tetgen.triangles = triangles
        tetgen.tetrahedra = baseTetra
        tetgen.verbosity = 0
        tetgen.max_tet_volume = 0.05
        tetgen.run()
        mesh = tetgen.mesh
        tetra = np.array(mesh.vertices)[np.array(mesh.elements)].transpose([1,0,2])
        scheme = quadpy.t3.keast_9()
        integrals = scheme.integrate(partial(self.rho, i = 0), tetra)
        return integrals.sum()

    def voronoi(self):
        if np.min(self.w) < 0:
            self.w += - np.min(self.w) * np.ones(self.w.shape)
        c = Container(self.X, limits = ((-self.a,-self.a,-self.a),(self.a,self.a,self.a)), radii = np.sqrt(self.w))
        for i in range(self.n):
            if c[i] is None:
                c[i] = EmptyCell(i, self.X[i,:])
        return c

    def voronoiOutOfState(self, X = None, w = None):
        if np.min(w) < 0:
            w += - np.min(w) * np.ones(w.shape)
        c = Container(X, limits = ((-self.a,-self.a,-self.a),(self.a,self.a,self.a)), radii = np.sqrt(w))
        for i in range(self.n):
            if c[i] is None:
                c[i] = EmptyCell(i, X[i,:])
        return c

    def eqTol(self,x,y,tol):
        if np.abs(x-y) < tol:
            return True
        return False

    def getFaces(self):
        facetsDict = {}
        for i in range(len(self.c)):
            neighbors = self.neighbors[i]
            if (neighbors == None).all():
                continue
            facet = self.facets[i]
            for k,j in enumerate(neighbors):
                if j >= 0:
                    key = tuple(sorted((i,j)))
                    facetsDict[key] = self.vertices[i][facet[k]]
        return facetsDict

    def cellsIntegral(self, f, c = None):
        if c is None:
            integrals = self.pool.imap(partial(self.integrateOverCell, f = f), enumerate(zip(self.vertices, self.facets)), chunksize = 15)
        else:
            integrals = self.pool.imap(partial(self.integrateOverCell, f = f), enumerate(zip(map(lambda x: np.array(x.vertices()), c), map(lambda x: x.face_vertices(), c))), chunksize = 15)
        return np.array(list(integrals))

    def facesIntegral(self):
        integrals = self.pool.imap(self.integrateOverFace, self.getFaces().items(), chunksize = 15)
        return np.array(list(integrals))

    def tetraVolume(self,points):
        newCol = np.ones((*points.shape[:-1],1))
        matrix = np.concatenate([points, newCol], axis = -1)
        return np.linalg.det(matrix) / 6

    def integrateOverCell(self, tup, f, scheme = quadpy.t3.keast_9()):
        i, tup2 = tup
        vertices, facets = tup2
        if (vertices == np.array(None)).all():
            if f(self.X[0].reshape(3,1,1),0).shape == (1,1):
                return float(0)
            else:
                return np.zeros(3)
        tetrahedrons = scipy.spatial.Delaunay(vertices).simplices
        volumes = self.tetraVolume(vertices[tetrahedrons])
        goodTetra = tetrahedrons[np.logical_not(np.isclose(volumes, 0, atol = 1e-11, rtol=1.e-5)),:]
        triangles = []
        for facet in facets:
            triangleMesh = pymesh.triangle()
            triangleMesh.verbosity = 0
            triangleMesh.points = np.array(vertices)[facet]
            triangleMesh.max_num_steiner_points = 0
            triangleMesh.run()
            triangles.append(np.array(facet)[triangleMesh.mesh.faces])
        triangles = np.vstack(triangles)
        tetgen = pymesh.tetgen()
        tetgen.points = vertices
        tetgen.verbosity = 0
        tetgen.triangles = triangles
        tetgen.max_tet_volume = 0.05
        tetgen.tetrahedra = goodTetra
        tetgen.run()
        mesh = tetgen.mesh
        tetra = np.array(mesh.vertices)[np.array(mesh.elements)]
        volumes = self.tetraVolume(tetra)
        goodTetra = tetra[np.logical_not(np.isclose(volumes, 0, atol = 1e-8, rtol = 1e-8)),:].transpose([1,0,2])
        integrals = scheme.integrate(partial(f, i = i), goodTetra)
        if len(integrals.shape) == 2:
            return integrals.sum(1)
        return integrals.sum(0)

    def integrateOverFace(self, tup, scheme = quadpy.t2.wandzura_xiao_6()):
        key, vertices = tup
        polygon = pyny.Polygon(vertices, False)
        param = polygon.get_parametric(True, tolerance=0.001)
        param /= np.linalg.norm(param[:3])
        i = np.argmax(np.abs(param[:3]))
        if self.eqTol(param[i],0,0.001):
            raise ArithmeticError('Face is not planar')
        reducedVertex = np.delete(vertices, i, axis = 1)
        triangles = scipy.spatial.Delaunay(reducedVertex).simplices
        triangleMesh = pymesh.triangle()
        triangleMesh.verbosity = 0
        triangleMesh.points = np.array(reducedVertex)
        triangleMesh.triangles = triangles
        triangleMesh.max_area = 0.05
        triangleMesh.run()
        mesh = triangleMesh.mesh
        integrals = scheme.integrate(partial(self.rhoface, param = param, i = i), mesh.vertices[mesh.elements].transpose([1,0,2]))
        return key, integrals.sum(0)

    def baricentres(self, c = None):
        num = self.cellsIntegral(self.xrho)
        den = np.copy(self.masses)
        num[np.where(den == 0)] = self.X[np.where(den == 0)[0],:]
        den[np.where(den == 0)] = 1
        bar = num / den.reshape(self.n,1)
        return bar

    def hessianW(self):
        integrals = self.facesIntegral()
        H = scipy.sparse.dok_matrix((self.n, self.n))
        for key, val in integrals:
            entry =  0.5 * (1 / np.linalg.norm(self.X[key[0],:] - self.X[key[1],:])) * val
            H[key] = H[key[::-1]] = - entry
            H[key[0], key[0]] += entry
            H[key[1], key[1]] += entry
        return -H

    def optimizeW(self, wCond, returnErrors = False, gamma = 0.001, max_iter = 20):
        errors = []
        Fs = []
        grad = self.m - self.masses
        errors.append(np.linalg.norm(grad))
        Fs.append(self.F())
        i = 0
        while not wCond(grad) and i < max_iter:
            H = self.hessianW()
            Hprimo = H.tocsc() - scipy.sparse.csc_matrix(gamma * np.eye(self.n))
            delta = scipy.sparse.linalg.spsolve(Hprimo, grad)
            if np.min(delta) > 0:
                delta -= np.min(delta)
            alpha = scipy.optimize.linesearch.line_search_armijo(lambda w: -self.FOutOfSTate(X = self.X, w = w), self.w, -delta, -grad, -self.F(), alpha0 = 10)[0]
            self.w -= alpha * delta
            self.c = self.voronoi()
            self.vertices = list(map(lambda x: np.array(x.vertices()), self.c))
            self.facets = list(map(lambda x: np.array(x.face_vertices()), self.c))
            self.neighbors = list(map(lambda x: np.array(x.neighbors()), self.c))
            self.masses = self.cellsIntegral(self.rho)
            grad = self.m - self.masses
            errors.append(np.linalg.norm(grad))
            Fs.append(self.F())
            i += 1
        if returnErrors:
            return errors, Fs

    def optimizeX(self):
        grad = (2 * self.masses.reshape(self.n, 1) * (self.X - self.baricentres()))
        minp = np.min((self.X[np.where(grad < 0)] - self.a) / grad[np.where(grad < 0)])
        minm = np.min((self.X[np.where(grad > 0)] + self.a) / grad[np.where(grad > 0)])
        alpha0 = np.min([minp, minm, self.alphabound]) - 1e-2
        alpha = scipy.optimize.linesearch.line_search_armijo(lambda X: self.FOutOfSTate(X = X, w = self.w), self.X.reshape(3 * self.n), -grad.reshape(3 * self.n), grad.reshape(3 * self.n), self.F(), alpha0 = alpha0)[0]
        self.X -= alpha * grad
        self.c = self.voronoi()
        self.vertices = list(map(lambda x: np.array(x.vertices()), self.c))
        self.facets = list(map(lambda x: np.array(x.face_vertices()), self.c))
        self.neighbors = list(map(lambda x: np.array(x.neighbors()), self.c))
        self.masses = self.cellsIntegral(self.rho)
        new_grad = (2 * self.masses.reshape(self.n, 1) * (self.X - self.baricentres()))
        maxMove = alpha * np.max(np.abs(grad))
        if maxMove > 5 * self.a * 1e-2:
            self.alphabound /= 2
            self.maxalpha /= 2
        if self.alphabound != self.maxalpha:
            if maxMove < self.a * 1e-2:
                if self.isBoundIncreasable:
                    self.alphabound *= 2
                    self.isBoundIncreasable = False
                    self.counterBound = 0
                else:
                    self.counterBound += 1
                    if self.counterBound == 5:
                        self.isBoundIncreasable = True
            else:
                self.counterBound = 0
        grad = new_grad

    def solve(self, wCond, XCond, verbose = False, returnErrors = False, tuning = True, max_iter = 500):
        """
            The solve function solves the 3D blue noise problem

            Parameters:

            self: object of class Optimizer
            wCond: function to be used as a criterion for convergence on the w-gradient
            XCond: function to be used as a criterion for convergence on the X-gradient
            verbose (default False): if True solve prints the state of the problem at each iteration
            returnErrors (default False): if True solve returns 3 lists:
                - list of 1-norm of the gradient at each iteration
                - list of 2-norm of the gradient at each iteration
                - list of the values of R(X, w) at each iteration
            tuning (default True): if False solve does not do the initial tuning (just for research purpose)
            max_iter (default 500): max number of iteration
        """
        errors1 = []
        errors2 = []
        costs = []
        if tuning:
            for _ in range(5):
                self.X = self.baricentres()
                self.c = self.voronoi()
                self.vertices = list(map(lambda x: np.array(x.vertices()), self.c))
                self.facets = list(map(lambda x: np.array(x.face_vertices()), self.c))
                self.neighbors = list(map(lambda x: np.array(x.neighbors()), self.c))
                self.masses = self.cellsIntegral(self.rho)
                self.grad[3 * self.n:] = self.m - self.masses
                self.grad[:3 * self.n] = (2 * self.masses.reshape(self.n, 1) * (self.X - self.baricentres())).reshape(self.n * 3)
                if verbose:
                    print("Finished X optimization, W-Grad 1-norm ->", np.max(np.abs(self.grad[3 * self.n:])))
                    print("Finished X optimization, X-Grad 2-norm: ", np.linalg.norm(self.grad[:3 * self.n]))
                self.optimizeW(wCond = wCond)
                self.grad[3 * self.n:] = self.m - self.masses
                self.grad[:3 * self.n] = (2 * self.masses.reshape(self.n, 1) * (self.X - self.baricentres())).reshape(self.n * 3)
                if verbose:
                    print("Finished W optimization, W-Grad 1-norm ->", np.max(np.abs(self.grad[3 * self.n:])))
                    print("Finished W optimization, X-Grad 2-norm: ", np.linalg.norm(self.grad[:3 * self.n]))
                    print("Cost value ->", self.cost())
        errors1.append(np.max(np.abs(self.grad)))
        errors2.append(np.linalg.norm(self.grad))
        costs.append(self.cost())
        if verbose:
            print('Tuning finished')
        while not wCond(self.grad[3 * self.n:]) or not XCond(self.grad[:3 * self.n]) and self.counter < max_iter:
            self.optimizeX()
            self.grad[3 * self.n:] = self.m - self.masses
            self.grad[:3 * self.n] = (2 * self.masses.reshape(self.n, 1) * (self.X - self.baricentres())).reshape(self.n * 3)
            if verbose:
                print("Finished X optimization, W-Grad 1-norm ->", np.max(np.abs(self.grad[3 * self.n:])))
                print("Finished X optimization, X-Grad 2-norm: ", np.linalg.norm(self.grad[:3 * self.n]))
            if wCond(self.grad[3 * self.n:]) and XCond(self.grad[:3 * self.n]):
                break
            self.optimizeW(wCond = wCond)
            self.grad[3 * self.n:] = self.m - self.masses
            self.grad[:3 * self.n] = (2 * self.masses.reshape(self.n, 1) * (self.X - self.baricentres())).reshape(self.n * 3)
            if verbose:
                print("Finished W optimization, W-Grad 1-norm ->", np.max(np.abs(self.grad[3 * self.n:])))
                print("Finished W optimization, X-Grad 2-norm: ", np.linalg.norm(self.grad[:3 * self.n]))
                print("Cost value ->", self.cost())
            costs.append(self.cost())
            errors1.append(np.max(np.abs(self.grad)))
            errors2.append(np.linalg.norm(self.grad))
            if self.counter >= 1:
                if np.max(np.abs(lastGrad - self.grad)) < 1e-8:
                    break
            lastGrad = np.copy(self.grad)
            self.counter += 1
        if returnErrors:
            return errors1, errors2, costs

    def saveNetwork(self, filename = 'network.obj', isOutside = None, intersect = None):
        """
            The saveNetwork function saves the diagram in an .obj file

            Parameters:

            self: object of class Optimizer
            filename (default 'network.obj'): path of the file to be saved
            isOutside (default None): function that accept a point of shape (1,3) and returns a Boolean
                Useful when the network needs to be intersected with a surface (Es for a cilynder isOutside(x) is x[0] ** 2 + x[1] ** 2 > r ** 2)
            intersect (default None): function that accepts two points x,y of shape (1, 3) where isOutside(x) is False and isOutside(y) is True
                and returns the intersection of the edge [x,y] with the surface.
        """
        nvertices = 0
        vertices = []
        edges = []
        for cell in self.c:
            vertices.append(np.array(cell.vertices()))
            for face in cell.face_vertices():
                for i in range(len(face)):
                    if i != len(face) - 1:
                        edges.append(np.array(sorted([nvertices + face[i], nvertices + face[i+1]])))
                    else:
                        edges.append(np.array(sorted([nvertices + face[i], nvertices + face[0]])))
            nvertices += len(cell.vertices())
        vertices = np.vstack(vertices)
        edges = np.unique(np.vstack(edges), axis = 0)
        if isOutside is None and intersect is None:
            network = pymesh.wires.WireNetwork.create_from_data(vertices, edges)
            network.write_to_file(filename)
        else:
            newVert = []
            newEdges = []
            for edge in edges:
                a = vertices[edge[0]]
                b = vertices[edge[1]]
                if isOutside(a) and isOutside(b):
                    pass
                elif isOutside(a) and not isOutside(b):
                    c = intersect(b,a)
                    newVert.append(c)
                    ic = len(newVert) - 1
                    if any(np.array_equal(b, x) for x in newVert):
                        ib = [np.array_equal(b, x) for x in newVert].index(True)
                    else:
                        newVert.append(b)
                        ib = ic + 1
                    newEdges.append(np.array([ib, ic]))
                elif isOutside(b) and not isOutside(a):
                    c = intersect(a,b)
                    newVert.append(c)
                    ic = len(newVert) - 1
                    if any(np.array_equal(a, x) for x in newVert):
                        ia = [np.array_equal(a, x) for x in newVert].index(True)
                    else:
                        newVert.append(a)
                        ia = ic + 1
                    newEdges.append(np.array([ia, ic]))
                else:
                    if any(np.array_equal(a, x) for x in newVert):
                        ia = [np.array_equal(a, x) for x in newVert].index(True)
                    else:
                        newVert.append(a)
                        ia = len(newVert) - 1
                    if any(np.array_equal(b, x) for x in newVert):
                        ib = [np.array_equal(b, x) for x in newVert].index(True)
                    else:
                        newVert.append(b)
                        ib = len(newVert) - 1
                    newEdges.append(np.array([ia, ib]))
            newVert = np.vstack(newVert)
            newEdges = np.unique(np.vstack(newEdges), axis = 0)
            network = pymesh.wires.WireNetwork.create_from_data(newVert, newEdges)
            network.write_to_file(filename)

    def quit(self):
        """
            The quit function shuts down the Pool object used for parallel integration
        """
        self.pool.close()
        self.pool.join()
