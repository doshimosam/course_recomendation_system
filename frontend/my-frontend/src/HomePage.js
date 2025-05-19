import React from "react";
import { Link } from "react-router-dom"; // Import Link from react-router-dom
import "./HomePage.css"; // External stylesheet for clean separation

const HomePage = () => {
  return (
    <div>
      {/* Navigation Bar */}
      <nav className="navbar">
        <Link to="/">
          <img src="img2.jpg" alt="Home Icon" /> Home
        </Link>
        <Link to="/Questionnaire">Questionnaire</Link>
      </nav>

      {/* College Header */}
      <section className="College">
        <div className="logo">
          <img src="SVKM Logo.png" alt="SVKM Logo" />
        </div>
        <h1>
          Shri Vile Parle Kelavani Mandal's <br />
          NARSEE MONJEE COLLEGE OF <br />
          COMMERCE & ECONOMICS (AUTONOMOUS)
        </h1>
      </section>

      {/* Hero Image */}
      <div className="img1">
        <img src="img1.jpg" alt="Career paths illustration" />
      </div>

      {/* Text Boxes */}
      <div className="text-box">
        Confused about what interests you and which course to select as per your interest?
      </div>
      <div className="text1-box">
        Here is the solution. Answer a few questions.
      </div>

      {/* Button */}
      <Link to="/Questionnaire" className="button1">
        Let's begin
      </Link>
    </div>
  );
};

export default HomePage;
