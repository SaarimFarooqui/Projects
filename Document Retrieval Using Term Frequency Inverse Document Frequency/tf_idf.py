document1 = "I have been playing video games since this morning. I am worried that spending too much time gaming all day might cause decline a in my grades. I am also very conscious about my mental well being and how much amount of sleep I am getting compared to what my body needs. I think it is difficult to divide a day properly between sleep, gaming, academic requirements and sports or physical activity. At most, one can effectively focus on two or maybe three of these."
document2 = "My favourite game is Grand Theft Auto V. Since I was a child, I have always loved the Grand Theft Auto series more than any other games. The best part of the game is the sense of freedom it offers in an open world. Another aspect that makes the game even more realistic is the presence of law enforcement. One of the best things ever happened to the Grand Theft Auto series was the development of Grand Theft Auto online which basically was released when the Grand Theft Auto 5 came out."
document3 = "Through online games, people can collectively solve large-scale computational problems. Such games constitute a general mechanism for using brain power to solve open problems. In fact, designing such a game is much like designing an algorithm - it must be proven correct, its efficiency can be analyzed, a more efficient version can supersede a less efficient one, and so on. Games with a purpose have a vast range of applications in areas as diverse as security, computer vision, Internet accessibility, adult content filtering, and Internet search."

documents = [document1, document2, document3]

def tf_idf(documents):
  merged_documents = ""
  for document in documents:
    document = document.lower() + " "
    merged_documents += document 
  return merged_documents

result = tf_idf(documents)

print(result)