#ifndef OFFSPRING_MAKER_H
#define OFFSPRING_MAKER_H

#include "Population.h"


class OffspringMaker
{
  public:
	virtual ~OffspringMaker() = default;
	virtual bool makeOffspring() = 0;
};

#endif // !OFFSPRING_MAKER_H
