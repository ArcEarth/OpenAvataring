inline void sincos(float v, __Out_Param__(float, s), __Out_Param__(float, c)) __GPU
{
	__MATH_NS sincos(v, &s, &c);
}

inline void sincos(float_2 v, __Out_Param__(float_2, s), __Out_Param__(float_2, c)) __GPU
{
	float sf, cf;
	__XX_SINCOSE_COMPOENT(x);
	__XX_SINCOSE_COMPOENT(y);
}

inline void sincos(float_3 v, __Out_Param__(float_3, s), __Out_Param__(float_3, c)) __GPU
{
	float sf, cf;
	__XX_SINCOSE_COMPOENT(x);
	__XX_SINCOSE_COMPOENT(y);
	__XX_SINCOSE_COMPOENT(z);
}

inline void sincos(float_4 v, __Out_Param__(float_4, s), __Out_Param__(float_4, c)) __GPU
{
	float sf, cf;
	__XX_SINCOSE_COMPOENT(x);
	__XX_SINCOSE_COMPOENT(y);
	__XX_SINCOSE_COMPOENT(z);
	__XX_SINCOSE_COMPOENT(w);
}