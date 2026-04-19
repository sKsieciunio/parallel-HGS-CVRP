#ifndef MIGRATION_POLICY_H
#define MIGRATION_POLICY_H

#include "../IslandState.h"

class MigrationPolicy {
public:
	virtual ~MigrationPolicy() = default;
	virtual bool shouldSend(const IslandState& islandState) = 0;
	virtual bool shouldReceive(const IslandState& islandState) = 0;
};

class FixedIntervalMigrationPolicy : public MigrationPolicy {
private:
	int interval;

public:
	FixedIntervalMigrationPolicy(int interval_) : interval(interval_) { }

	bool shouldSend(const IslandState& islandState) override {
		return islandState.iteration % interval == 0;
	}

	bool shouldReceive(const IslandState& islandState) override {
		return islandState.iteration % interval == 0;
	}
};

class ImprovementTriggeredMigrationPolicy : public MigrationPolicy {
private:
    int warmup;
    int sendCooldown;
    int receiveStagnationThreshold;
    int lastSendIteration = -1000000;

public:
    ImprovementTriggeredMigrationPolicy(
        int warmup, int sendCooldown, int receiveStagnationThreshold)
        : warmup(warmup)
        , sendCooldown(sendCooldown)
        , receiveStagnationThreshold(receiveStagnationThreshold) {
    }

    bool shouldSend(const IslandState& state) override {
        if (state.iteration < warmup) return false;
        if (!state.foundNewBest) return false;
        if (state.iteration - lastSendIteration < sendCooldown) return false;
        lastSendIteration = state.iteration;
        return true;
    }

    bool shouldReceive(const IslandState& state) override {
        if (state.iteration < warmup) return false;
        return state.iterationWithoutImprovement >= receiveStagnationThreshold;
    }
};

class AdaptiveMigrationPolicy : public MigrationPolicy {
private:
    int sendCooldown;
    int minReceiveInterval;
    int maxReceiveInterval;
    int warmup;
    int lastSendIteration = -1000000;
    int lastReceiveCheckIteration = -1000000;

public:
    AdaptiveMigrationPolicy(
        int sendCooldown, int minReceiveInterval,
        int maxReceiveInterval, int warmup)
        : sendCooldown(sendCooldown)
        , minReceiveInterval(minReceiveInterval)
        , maxReceiveInterval(maxReceiveInterval)
        , warmup(warmup) {
    }

    int currentReceiveInterval(const IslandState& state) const {
        double ratio = (double)state.iterationWithoutImprovement / state.maxIterNoImprovement;
        return (int)(maxReceiveInterval - ratio * (maxReceiveInterval - minReceiveInterval));
    }

    bool shouldSend(const IslandState& state) override {
        if (state.iteration < warmup) return false;
        if (!state.foundNewBest) return false;
        if (state.iteration - lastSendIteration < sendCooldown) return false;
        lastSendIteration = state.iteration;
        return true;
    }

    bool shouldReceive(const IslandState& state) override {
        if (state.iteration < warmup) return false;
        int interval = currentReceiveInterval(state);
        if (state.iteration - lastReceiveCheckIteration >= interval) {
            lastReceiveCheckIteration = state.iteration;
            return true;
        }
        return false;
    }
};

#endif // !MIGRATION_POLICY_H
