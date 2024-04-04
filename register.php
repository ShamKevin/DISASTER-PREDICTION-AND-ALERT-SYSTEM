<?php
// Establish database connection (replace placeholders with actual database credentials)
$servername = "localhost";
$username = "your_username";
$password = "your_password";
$database = "your_database";

$conn = new mysqli($servername, $username, $password, $database);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Process form data when form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = $_POST["name"];
    $mobile = $_POST["mobile"];
    $otp = $_POST["otp"];
    $password = $_POST["password"];
    $confirm_password = $_POST["confirm-password"];

    // Perform necessary validation here

    // Insert data into the database
    $sql = "INSERT INTO users (name, mobile, otp, password) VALUES ('$name', '$mobile', '$otp', '$password')";

    if ($conn->query($sql) === TRUE) {
        echo "Registration successful!";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

// Close database connection
$conn->close();
?>
