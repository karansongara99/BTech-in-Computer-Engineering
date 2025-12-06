function personal(name, age) {
    console.log("This is the personal Information.");
}

function details(city, country, callback) {
    console.log("City:", city);
    console.log("Country:", country);
    callback();
}
details("New York", "USA", function() {
    personal("Alice", 30);
}
);