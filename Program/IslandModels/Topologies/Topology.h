#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include <vector>

class Topology {
protected:
    int n_islands;
    int island_idx;
    std::vector<int> neighbors;

public:
    virtual ~Topology() = default;
    const std::vector<int>& getNeighbors() const { return neighbors; }

    Topology(int n_islands_, int island_idx_)
        : n_islands(n_islands_), island_idx(island_idx_) {
    }
};

class RingTopology : public Topology {
public:
    RingTopology(int n_islands_, int island_idx_)
        : Topology(n_islands_, island_idx_) {
        neighbors = { (island_idx + 1) % n_islands };
    }
};

class TwoSidedRing : public Topology {
public:
    TwoSidedRing(int n_islands_, int island_idx_)
        : Topology(n_islands_, island_idx_) {
        neighbors = { (island_idx + 1) % n_islands, (n_islands + island_idx - 1) % n_islands };
    }
};

class Hypercube : public Topology {
public:
    Hypercube(int n_islands_, int island_idx_) :
        Topology(n_islands_, island_idx_) {
        neighbors = {};
        for (int i = 0; i < n_islands; i <<= 1) {
            neighbors.push_back(i ^ island_idx);
        }
    }
};

class StarTopology : public Topology {
public:
    StarTopology(int n_islands_, int island_idx_)
        : Topology(n_islands_, island_idx_)
    {
        if (island_idx == 0) {
            for (int i = 1; i < n_islands; i++) {
                neighbors.push_back(i);
            }
        }
        else {
            neighbors.push_back(0);
        }
    }
};

class FullGraphTopology : public Topology {
public:
    FullGraphTopology(int n_islands_, int island_idx_)
        : Topology(n_islands_, island_idx_)
    {
        for (int i = 0; i < n_islands; i++) {
            if (i != island_idx) {
                neighbors.push_back(i);
            }
        }
    }
};

#endif