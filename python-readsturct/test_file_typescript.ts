async function fetchData(url: string): Promise<any> {
  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`Error: ${response.status} - ${response.statusText}`)
    }
    const data = await response.json()
    return data
  } catch (error) {
    console.error('Fetch failed:', error)
    throw error // Rethrow the error for further handling
  }
}
function merge<T extends object, U extends object>(obj1: T, obj2: U): T & U {
    return { ...obj1, ...obj2 };
}

function processValue(value: string | number): string {
    if (typeof value === 'string') {
        return `String value: ${value.toUpperCase()}`;
    } else if (typeof value === 'number') {
        return `Number value: ${value * 2}`;
    }
    return 'Invalid value';
}

console.log(processValue('hello')); // Output: String value: HELLO
console.log(processValue(10));       // Output: Number value: 20

function factorial(n: number): number {
    if (n < 0) {
        throw new Error('Factorial is not defined for negative numbers.');
    }
    if (n === 0 || n === 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

console.log(factorial(5)); // Output: 120


const merged = merge({ name: 'Alice' }, { age: 30 });
console.log(merged); // Output: { name: 'Alice', age: 30 }

// Usage
fetchData('https://api.example.com/data')
  .then((data) => console.log(data))
  .catch((err) => console.error('Error fetching data:', err))


