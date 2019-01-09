# Common Regular Expressions

## Matching a Username

> /^[a-z0-9_-]{3,16}$/

The beginning of the string (^), followed by any lowercase letter (a-z), number (0-9), an underscore, or a hyphen. Next, {3,16} makes sure that are at least 3 of those characters, but no more than 16. Finally, the end of the string ($)

## Matching a Hex Value

> /^#?([a-f0-9]{6}|[a-f0-9]{3})$/

The beginning of the string (^). Next, a (#) is optional because it is followed by a (?). The question mark signifies that the preceding character is optional, but to capture it if it's there. Next, inside the first group (first group of parentheses), we can have two different situations. The first is any lowercase letter between a and f or a number six times. The vertical bar tells us that we can also have three lowercase letters between a and f or numbers instead. Finally, we want the end of the string ($).


## Matching an Email  

> /^([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$/
 
The beginning of the string (^). Inside the first group, match one or more lowercase letters, numbers, underscores, dots, or hyphens. The dot is escaped because a non-escaped dot means any character. Directly after that, there must be an at sign. Next is the domain name which must be: one or more lowercase letters, numbers, underscores, dots, or hyphens. Then another (escaped) dot, with the extension being two to six letters or dots. There are 2 to 6 because of the country specific TLD's (.com or .co.in). Finally, we want the end of the string ($).
  
  
## Matching a URL  

> /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/

The first capturing group is all option. It allows the URL to begin with "http://", "https://", or neither of them. A question mark after the s allows URL's that have http or https. In order to make this entire group optional, a question mark is added to the end of it.

Next is the domain name: one or more numbers, letters, dots, or hypens followed by another dot then two to six letters or dots. The following section is the optional files and directories. Inside the group, match is made for any number of forward slashes, letters, numbers, underscores, spaces, dots, or hyphens. Then this group is allowed to be matched as many times as we want. Pretty much this allows multiple directories to be matched along with a file at the end. A star is used instead of the question mark because the star says zero or more, not zero or one. If a question mark was to be used there, only one file/directory would be able to be matched.

Then a trailing slash is matched, but it can be optional. Finally end with the end of the line.


## Matching an HTML Tag 

> /^<([a-z]+)([^<]+)*(?:>(.*)<\/\1>|\s+\/>)$/

It matches any HTML tag with the content inside. As usual, begin with the start of the line.

First comes the tag's name. It must be one or more letters long. The next thing are the tag's attributes. This is any character but a greater than sign (>). Star is used since its optional and may have more than 1 character. The plus sign makes up the attribute and value, and the star says as many attributes as you want.

Next comes the third non-capture group. Inside, it will contain either a greater than sign, some content, and a closing tag; or some spaces, a forward slash, and a greater than sign. The first option looks for a greater than sign followed by any number of characters, and the closing tag. \1 is used which represents the content that was captured in the first capturing group. In this case it was the tag's name. Now, if that couldn't be matched we want to look for a self closing tag (like an img, br, or hr tag). This needs to have one or more spaces followed by "/>".

The regex is ended with the end of the line.


## Matching an IP Address 

 > /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/
