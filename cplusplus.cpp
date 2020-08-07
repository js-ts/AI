// This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iomanip>
using namespace std;
using std::cin;
using std::cout;
using std::endl;
using std::setw;

#define PI 3.1415926
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define ToStr(x) #x
#define Concat(x, y) x##y
#define PRINT(x) (cout << x << endl) 

#ifdef DEBUG
    #define A 10
#endif // DEBUG

int A = 10;

namespace TEST {
    int A = 20;
    void print() {
        cout << A << endl;
    }

    template <typename T>
    void print(T x, T y) {
        cout << x << y << endl;
    }
}

int ADD(int a, int b)
{
    return a + b;
}

// 
template <typename T>
inline T const& MAX (T const& a, T const& b) {
    return a > b ? a : b;
}


// 
template <typename T>
void SWAP(T& a, T& b) {  // notice  do not use (T a, T b)
    T t;
    t = a;
    a = b;
    b = t;
    return;
}

// 
template <typename T>
bool cmp(T& a, T& b) {
    return a > b;
}


//  class
template <typename T>
class STACK {
private:
    vector<T> vec;
public:
    void push(T const&);
    void pop();
    T top() const;
    bool empty() const { return vec.empty(); }
};

//
template <typename T>
void STACK<T>::push(T const& e) {
    vec.push_back(e);
}

template <typename T>
void STACK<T>::pop() {
    if (vec.empty()) {
        throw out_of_range("---");
    }
    vec.pop_back();
}

template <typename T>
T STACK<T>::top() const {
    if (vec.empty()) {
        throw out_of_range("----");
    }
    return vec.back();
    //return vec[-1];
}


// class
// pure /  virtual function
class BoxParent {
protected:
    double l;
    double w;
    double b;
public:
    //virtual double foobar() = 0; //interface
    virtual double foobar() { return 0.0; }
};

class Box : public BoxParent{

public:
    //Box() : l(0), w(0), b(0) { ; };
    Box() {};
    Box(double, double, double);
    ~Box() { ; }

    void setL(double l) { this->l = l; }
    void setW(double w) { this->w = w; }
    void setB(double b) { this->b = b; }
    double getVolume() const { return l * w * b; } // const 
    double getAera();
    bool compare(const Box&);

    Box operator+(const Box&);
    Box operator-();
    bool operator>(const Box&);

    double foobar() { return 100000.0; }

//private:
//    double l;
//    double w;
//    double b;
};

Box::Box(double a, double b, double c) {
    this->l = a;
    this->w = b;
    this->b = c;
}

double Box::getAera() {
    return w * b;
}

bool Box::compare(const Box& box) {
    return this->getVolume() >= box.getVolume();
}

Box Box::operator+(const Box& other) {
    Box box;
    box.l = this->l + other.l;
    box.w = this->w + other.w;
    box.b = this->b + other.b;
    return box;
}
Box Box::operator-() {
    this->l = -l;
    this->w = -w;
    this->b = -b;
    return Box(l, w, b);
}

bool Box::operator>(const Box& other) {
    if (this->getVolume() > other.getVolume()) { // const object can only call const method
        return true;
    }
    return false;
}

// 
class utils
{
public:

    static void print(int x) { cout << x << endl; }
    static void print(double x) { cout << x << endl; }
    static void print(string x) { cout << x << endl; }
    static void print(char x[]) { cout << x << endl; }

private:
    ;
};




int main()
{
    // 
    std::cout << "Hello World!\n";
    std::cout << ADD(10, 20) << std::endl;

    // type
    cout << numeric_limits<int>::max() << endl;
    cout << sizeof(int) << endl;

    // enum
    enum Color { blue, red};
    Color color = Color(1);
    cout << color << endl;

    // template macro
    cout << MAX(10, 20) << endl;
    cout << MAX(10.2, 6.5) << endl;
    cout << MIN(10.1, 40) << endl;

    // string
    string s;
    string stra = "stra";
    string strb = "strb";
    cout << s + stra + strb << setw(5) << (stra + strb).size() << endl;

    // vector
    vector<int> vec;
    for (int i = 0; i < 5; i++) {
        vec.push_back(i);
    }

    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i] << endl;
    }

    //vector<int>::iterator iter = vec.begin()
    for (auto iter = vec.begin(); iter != vec.end(); iter++) {
        cout << *iter << endl;
    }

    // algorithm
    while (next_permutation(vec.begin(), vec.begin()+3))
    {
        cout << vec[0] << vec[1] << vec[2] << endl;
    }

    sort(vec.begin(), vec.end(), cmp<int>);
    reverse(vec.begin(), vec.end());
    for (auto a : vec) {
        cout << a << endl;
    }

    // template
    STACK<int> stack;
    stack.push(10);
    stack.push(20);
    stack.pop();
    cout << stack.top() << endl;

    // macro
    cout << min(10., 10.2) << max(22, 22) << endl;
    
    int ab = 1000;
    cout << ToStr(dasf) << "  " << Concat(a, b) << endl;

    cout << ::A << endl;
    cout << TEST::A << endl;
    TEST::print();

    PRINT(ab);


    // cmath
    cout << cos(PI / 2.0) << endl;

    // 
    cout << __LINE__ << endl;
    cout << __FILE__ << endl;
    cout << __DATE__ << endl;
    cout << __TIME__ << endl;

    // arr pointer
    int arr[10] = { 1, 10, 3 };
    int* ptr = arr;
    for (auto a : arr) {
        cout << ptr << setw(5) << a << endl;
        ptr++;

    }

    // class
    BoxParent* boxp;
    Box box, box1, box2;
    box.setB(10);
    box.setL(30);
    box.setW(40);
    box1 = box + box;
    box2 = -box;
    boxp = &box2;

    PRINT(box.getVolume());
    PRINT(box1.getVolume());
    utils().print(box2.getVolume());
    utils().print(box2 > box);
    utils().print(boxp->foobar());
    if (box.compare(box)) {
        cout << "larger or equal than" << endl;
    }

    // static
    utils().print(10);
    utils::print(10.0);

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
