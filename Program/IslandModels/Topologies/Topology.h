#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include <vector>
#include <set>
#include <string>
#include <random>
#include <algorithm>

class Topology 
{
protected:
    int n_islands;
    int island_idx;
    std::vector<int> neighbors;

public:
    virtual ~Topology() = default;
    const std::vector<int>& getNeighbors() const { return neighbors; }

    Topology(int n_islands_, int island_idx_)
        : n_islands(n_islands_), island_idx(island_idx_) 
    {
    }
};

class RingTopology : public Topology 
{
public:
    RingTopology(int n_islands_, int island_idx_)
        : Topology(n_islands_, island_idx_) 
    {
        neighbors = { (island_idx + 1) % n_islands };
    }
};

class TwoSidedRing : public Topology 
{
public:
    TwoSidedRing(int n_islands_, int island_idx_)
        : Topology(n_islands_, island_idx_) 
    {
        neighbors = { (island_idx + 1) % n_islands, (n_islands + island_idx - 1) % n_islands };
    }
};

class Hypercube : public Topology
{
public:
    Hypercube(int n_islands_, int island_idx_) :
        Topology(n_islands_, island_idx_)
    {
        if (n_islands_ > 1 && (n_islands_ & (n_islands_ - 1)) != 0)
            throw std::string("Hypercube topology requires nbNodes to be a power of 2 (got "
                + std::to_string(n_islands_) + "). Use -topology 5 (RandomRegular) instead.");
        neighbors = {};
        for (int i = 1; i < n_islands; i <<= 1)
        {
            neighbors.push_back(i ^ island_idx);
        }
    }
};

class StarTopology : public Topology 
{
public:
    StarTopology(int n_islands_, int island_idx_)
        : Topology(n_islands_, island_idx_)
    {
        if (island_idx == 0) 
        {
            for (int i = 1; i < n_islands; i++)
            {
                neighbors.push_back(i);
            }
        }
        else 
        {
            neighbors.push_back(0);
        }
    }
};

class FullGraphTopology : public Topology
{
public:
    FullGraphTopology(int n_islands_, int island_idx_)
        : Topology(n_islands_, island_idx_)
    {
        for (int i = 0; i < n_islands; i++)
        {
            if (i != island_idx)
            {
                neighbors.push_back(i);
            }
        }
    }
};

class RandomRegularTopology : public Topology
{
public:
    RandomRegularTopology(int n_islands_, int island_idx_, int degree, unsigned seed)
        : Topology(n_islands_, island_idx_)
    {
        std::vector<std::vector<int>> adjacency = buildAdjacency(n_islands_, degree, seed);
        neighbors = adjacency[island_idx_];
    }

private:
    static std::vector<std::vector<int>> buildAdjacency(int n, int k, unsigned seed)
    {
        if (k > n - 1) k = n - 1;
        if (n <= 1 || k <= 0) return std::vector<std::vector<int>>(std::max(n, 0));
        if ((n * k) % 2 != 0) k -= 1;
        if (k <= 0) return std::vector<std::vector<int>>(n);

        std::mt19937 rng(seed);
        const int maxAttempts = 1000;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            std::vector<int> stubs;
            stubs.reserve((size_t)n * k);
            for (int v = 0; v < n; v++)
                for (int j = 0; j < k; j++)
                    stubs.push_back(v);
            std::shuffle(stubs.begin(), stubs.end(), rng);

            std::vector<std::set<int>> adj(n);
            bool ok = true;
            for (size_t i = 0; i + 1 < stubs.size(); i += 2)
            {
                int a = stubs[i], b = stubs[i + 1];
                if (a == b || adj[a].count(b)) { ok = false; break; }
                adj[a].insert(b);
                adj[b].insert(a);
            }
            if (!ok) continue;
            return toVectors(adj);
        }

        return circulantAdjacency(n, k);
    }

    static std::vector<std::vector<int>> circulantAdjacency(int n, int k)
    {
        std::vector<std::set<int>> adj(n);
        int half = k / 2;
        for (int v = 0; v < n; v++)
            for (int d = 1; d <= half; d++)
            {
                adj[v].insert((v + d) % n);
                adj[v].insert((v - d + n) % n);
            }
        if (k % 2 == 1 && n % 2 == 0)
            for (int v = 0; v < n; v++)
                adj[v].insert((v + n / 2) % n);
        return toVectors(adj);
    }

    static std::vector<std::vector<int>> toVectors(const std::vector<std::set<int>>& adj)
    {
        std::vector<std::vector<int>> result(adj.size());
        for (size_t v = 0; v < adj.size(); v++)
            result[v].assign(adj[v].begin(), adj[v].end());
        return result;
    }
};

#endif