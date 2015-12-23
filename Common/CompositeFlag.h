#pragma once

template <typename _Tenum = unsigned int>
class CompositeFlag
{
public:
	static const CompositeFlag All() {
		return CompositeFlag(all);
	}
	static const CompositeFlag None() {
		return CompositeFlag(none);
	}

	CompositeFlag() = default;

	CompositeFlag(unsigned int flag)
	{
		Flag = flag;
	}

	CompositeFlag(int flag)
	{
		Flag = (unsigned)flag;
	}

	CompositeFlag(_Tenum flag)
	{
		Flag = flag;
	}

	unsigned int& RawData() { return Flag; }

	void ClearAll() { Flag = none;}
	void SetAll() {Flag = all;}

	void Set(_Tenum Key) 
	{ 
		Flag |= 1<<Key;
	}

	void Clear(_Tenum Key)
	{ 
		Flag &= all ^ (1<<Key);
	}

	void Toggle(_Tenum Key)
	{
		Flag ^= 1<<Key;
	}

	void Specify(_Tenum Key)
	{
		Flag = 1<<Key;
	}

	unsigned int Contains(_Tenum Key) 
	{
		return (Flag & 1<<Key);
	}
	unsigned int Contains(const CompositeFlag& rhs) 
	{
		return (Flag == rhs.Flag) || ((Flag & (Flag ^ rhs.Flag)) == (Flag ^ rhs.Flag));
	}

	CompositeFlag& operator+= (_Tenum Key) 
	{
		Flag |= 1<<Key;
		return *this;
	}
	CompositeFlag& operator-= (_Tenum Key) 
	{
		Flag &= all ^ (1<<Key);
		return *this;
	}
	CompositeFlag& operator^= (_Tenum Key) 
	{
		Flag ^= 1<<Key;
		return *this;
	}
	//CompositeFlag& operator= (unsigned int Key)
	//{
	//	Flag = 1<<Key;
	//}

	unsigned int operator>= (_Tenum Key)
	{
		return (Flag & 1<<Key);
	}

	unsigned int operator>= (const CompositeFlag& rhs)
	{
		return (Flag & rhs.Flag);
	}

	CompositeFlag& operator= (const CompositeFlag& rhs)
	{
		Flag = rhs.Flag;
		return *this;
	}
	CompositeFlag& operator+= (const CompositeFlag& rhs)
	{
		Flag |= rhs.Flag;
		return *this;
	}
	CompositeFlag& operator-= (const CompositeFlag& rhs)
	{
		Flag &= rhs.Flag;
		return *this;
	}
	CompositeFlag& operator^= (const CompositeFlag& rhs)
	{
		Flag ^= rhs.Flag;
		return *this;
	}

private:
	unsigned int Flag; 
	static const unsigned int all = 0xffffffff;
	static const unsigned int none = 0;
};
