## define your few-shot example prompt
## TODO: make it with txt path, maybe..

fewshot_prompt = """
Below are example conversations to guide the flow and structure of your discussion:

Example 1:
User A: The project deadline is on the 15th of next month. What do we need to do?  
User B: First, we should set the schedule for design and development. Development can only start once the design is finished.  
User C: I’ll take care of the testing schedule. I think we can start right after development is done.  
User A: Great, then how about we allocate one week for design, two weeks for development, and one week for testing?  
User B: Sounds good. Let’s ask AI to organize the overall schedule.  
User C: Sure, I’ll ask AI.  
User A: Then, who will take care of the design?  
User B: I’ll do it. I handled it last time, so I think it’ll be easier if I do it again this time.  
User C: I can handle the development. I’ll start as soon as the design is completed, right?  
User A: Yes, and I’ll take on the task of compiling the test results and managing the entire project.  
User C: Perfect. Now that we’ve divided the roles, we just need to ask AI to organize everything.  
User C: AI, please organize the project’s design, development, and testing schedules. Refer to our conversation to update the calendar and the project management information as well.  

Example 2:  
User A: We need to check the sales data for this quarter.  
User B: Right. It’ll be an important benchmark for the upcoming meeting.  
User C: Just looking at sales isn’t enough. It would be great if we could also have data comparing us to our competitors.  
User A: Exactly. Comparing with competitors will help us understand where we’re doing better.  
User B: It’d also be helpful to look at market share. We need to see where we stand in the overall market.  
User C: Good idea. Let’s ask AI to fetch real-time data. I’ll make the request.  
User C: AI, please fetch the sales data for this quarter, competitor data, and market share data.  

Example 3:
User A: Shall we have dinner together this weekend? How about Korean food?
User B: Sounds good, but I’m vegetarian, so meat-heavy menus are a bit difficult for me. We should look for a place that offers vegetarian options.
User C: I’d prefer a place with easy parking. These days, restaurants without parking spaces are really inconvenient.
User A: How about near Gangnam Station? It’s about the same distance from everyone’s place, and there are plenty of Korean restaurants there.
User B: That sounds good. We should be able to find a place with decent vegetarian options near Gangnam Station.
User C: There are many places near Gangnam Station with parking, but a restaurant with its own parking lot would be even better.
User A: How about 6 PM for dinner? If it’s too late, it might be hard to get a table.
User B: That works for me. 6 PM is fine.
User C: So, our requirements are Korean food, vegetarian options, and parking availability.
User A: That’s right. Let’s ask AI. I’ll make the request.
User A: AI, please find a Korean restaurant near Gangnam Station for 6 PM this weekend that offers vegetarian options. It’d be great if it has parking as well. Please provide a list of recommendations.
"""
