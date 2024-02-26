#pragma once

#include <memory>

template <typename T>
class MyVector final {

    static constexpr size_t defaultCapacity = 8;
  
  private:
    size_t size{};
    size_t cap{};
    std::shared_ptr<T[]> arr{};

    void expand(size_t newCap);

  public:
    MyVector() = default;
    MyVector(size_t _size, T value);
    explicit MyVector(size_t _size);
    MyVector(const std::initializer_list<T>& list);

    MyVector(const MyVector<T>& other);
    MyVector(MyVector<T>&& other) noexcept;
    MyVector<T>& operator=(const MyVector& other);
    MyVector<T>& operator=(MyVector<T>&& other) noexcept;
    ~MyVector() = default;

	void push_back(const T& elem);
    void pop_back();
    void insert(size_t index, const T& elem);
    void erase(size_t index);
    [[nodiscard]] size_t length() const noexcept;
    [[nodiscard]] size_t capacity() const noexcept;

    T& operator[](size_t index) const;

    [[nodiscard]] T* begin() const noexcept;
    [[nodiscard]] T* end() const noexcept;
};

template<class T>  
void MyVector<T>::expand(size_t newCap) {
    std::shared_ptr<T[]> newArr;

    if(newCap <= cap) {
    return;
    }

    newArr = std::shared_ptr<T[]>(new T[newCap]);
    for(size_t i = 0; i < size; ++i) {
        newArr[i] = arr[i];
    }

    arr = newArr;
    cap = newCap;
}   

template<class T> 
MyVector<T>::MyVector(size_t _size, T value): size(_size), cap(size), arr(std::shared_ptr<T[]>(new T[cap])) {
    for(int i = 0; i < size; ++i) {
        arr[i] = value;
    }
}

template<class T> 
MyVector<T>::MyVector(size_t _size): size(_size), cap(size > 0 ? 2*size : defaultCapacity), arr(std::shared_ptr<T[]>(new T[cap])) {}

template <class T> 
MyVector<T>::MyVector(const std::initializer_list<T>& list): size(list.size()), cap(size > 0 ? 2*size : defaultCapacity), arr(std::shared_ptr<T[]>(new T[cap])) {
    int i = 0;
    for(const T& elem : list) {
        arr[i++] = elem;
    }
}

// Правило пяти
template <class T> 
MyVector<T>::MyVector(const MyVector<T>& other): size(other.size), cap(other.cap), arr(std::shared_ptr<T[]>(new T[cap])) {
    for(int i = 0; i < size; ++i) {
        arr[i] = other[i];
    }
}

template <class T>
MyVector<T>::MyVector(MyVector<T>&& other) noexcept : size(other.size), cap(other.cap), arr(other.arr) {
    other.arr = nullptr;
    other.size = 0;
    other.cap = 0;
}

template <class T>
MyVector<T>& MyVector<T>::operator=(const MyVector& other){
    size = other.size;
    cap = other.cap;
    arr = std::shared_ptr<T[]>(new T[cap]);
    for(int i = 0; i < size; ++i) {
        arr[i] = other[i];
    }
    return *this;
}

template <class T> 
MyVector<T>& MyVector<T>::operator=(MyVector<T>&& other) noexcept {
    size = other.size;
    cap = other.cap;
    arr = other.arr;

    other.arr = nullptr;
    other.size = 0;
    other.cap = 0;
    return *this;
}

// Методы
template <class T> 
void MyVector<T>::push_back(const T& elem)  {

    if(size < cap) {
        arr[size++] = elem;
    } else {
        this -> expand(cap > 0 ? 2*cap : defaultCapacity);
        arr[size++] = elem;
    }
}

template <class T> 
void MyVector<T>::pop_back()  {

    if(size == 0) {
        throw std::logic_error("Cannot pop_back() element from empty MyVector");
    }

    --size;
}

template <class T> 
void MyVector<T>::insert(size_t index, const T& elem)  {
    if(index > size) {
        throw std::logic_error("Cannot insert Element on this position: index out of range");
    }

    if(size == cap) {
        this -> expand(cap > 0 ? 2*cap : defaultCapacity);
    }

    ++size;
    for(int i = size - 2; i >= index; --i) {
        arr[i+1] = arr[i]; 
    }
    arr[index] = elem;
}

template <class T> 
void MyVector<T>::erase(size_t index)  {
    
    if(size == 0) {
        throw std::logic_error("Cannot pop_back() element from empty MyVector");
    } else if (index >= size) {
        throw std::logic_error("cannot erase from this position: index out of rage");
    } else {
        for(int i = index + 1; i < size; ++i) {
            arr[i-1] = arr[i];
        }
        --size;
    }
}

template<class T> 
size_t MyVector<T>::length() const noexcept  {
    return size;
}

template<class T> 
size_t MyVector<T>::capacity() const noexcept  {
    return cap;
}

// Оператор []
template <class T> 
T& MyVector<T>::operator[](size_t index) const  {
    if(index >= size) {
        throw std::logic_error("Cannot take Element from this index: index out of range");
    }

    return arr[index];
}

// Для итераторов
template <class T> 
T* MyVector<T>::begin() const noexcept  {
    return arr.get();
}

template <class T> 
T* MyVector<T>::end() const noexcept  {
    return arr.get() + size;
}
