class classobject
{
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    display() {
        console.log("Name:", this.name);
        console.log("Age:", this.age);
    }
}
let person1 = new classobject("Karan", 20);
person1.display();