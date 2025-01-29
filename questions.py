from collections import defaultdict

list = [{"name":"a","category":"Normal", "price":1},{"name":"b","category":"Special", "price":1},{"name":"c","category":"Special", "price":1},{"name":"d","category":"Normal", "price":1}]



# list3 = {"Special":[c,b],"Normal":[a,d]} 



def categorize_items(items):
    categorized_items = defaultdict(list)
    for item in items:
        categorized_items[item['category']].append(item['name'])
    return categorized_items

list3 = categorize_items(list)
print(list3)





    



numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def separate_even_odd(numbers):
    even = [num for num in numbers if num % 2 == 0]
    odd = [num for num in numbers if num % 2 != 0]
    return {'even': even, 'odd': odd}

result = separate_even_odd(numbers)
print(result)
        
     

    

        