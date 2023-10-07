import matplotlib.pyplot as plt

# Summary of the numbers
categories = ['Financial Company Approvals', 'Bank Institutions', 'Continuing Financial Products', 'Net Value Financial Products', 'Investment Nature', 'Investor Growth', 'Sales Channels']
counts = [31, 278, 3.47, 26.4, 94.5, 9671, 328]

# Plotting a bar chart
plt.figure(figsize=(10, 6))
plt.bar(categories, counts, color='skyblue')
plt.title('Summary of Bank Wealth Management Market Development')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.show()

# Plotting a pie chart
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightpink', 'lightsalmon', 'lightyellow', 'lightcoral', 'lightcyan'])
plt.title('Summary of Bank Wealth Management Market Development')
plt.show()