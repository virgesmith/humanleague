
#pragma once

#include <memory>
#include <mutex>

// Threadsafe(?) global singleton access
namespace Global
{
	template<typename T>
	T& instance()
	{
	  static std::unique_ptr<T> instance;
		static std::once_flag init;
		std::call_once(init, [](){ instance.reset(new T); });
		return *instance;
	}
}
