#include <iostream>
#include <memory>

void printarray(int *arr, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        std::cout << "Value at index " << i << ": " << arr[i] << std::endl;
        /* code */
    }
}
int main()
{
    // std::unique_ptr<int[]> arr(new int[5]);
    // for (unsigned char i = 0; i < 5; i++)
    // {
    //     arr[i] = i * 2;
    //     std::cout << "Value at index " << static_cast<int>(i) << ": " << arr[i] << std::endl;
    // }

    // int a[] = {1, 2, 5};
    // int b[3] = {3, 4, 6};

    // int sum[3];
    // for (unsigned char i = 0; i < 3; i++)
    // {
    //     sum[i] = a[i] + b[i];
    //     std::cout << "Sum of elements at index " << static_cast<int>(i) << std::endl;
    // }
    int b[] = {0, 1, 2, 3, 4, 5};
    printarray(b, sizeof(b) / sizeof(b[0]));
    return 0;
}