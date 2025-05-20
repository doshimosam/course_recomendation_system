
import React from "react";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import './Questionnaire.css';


function Questionnaire() {
  const [questions, setQuestions] = useState([]);
  const [responses, setResponses] = useState({});
  const navigate = useNavigate();

//   useEffect(() => {
//   axios.get("http://127.0.0.1:5000/questions").then((response) => {
//     setQuestions(response.data);
//   });
// }, []);


  useEffect(() => {
    axios.get("https://course-recommendation-system-b1wi.onrender.com/questions").then((response) => {
      setQuestions(response.data);
    });
  }, []);

  const handleSubmit = () => {
    const answers = questions.map((q, index) => responses[index] || "");
    axios
      .post("https://course-recommendation-system-b1wi.onrender.com/recommend", { responses: answers })
      .then((response) => {
        navigate("/result", {
          state: {
            recommended_course: response.data.recommended_course,
            description: response.data.description
          }
        });
      })
      .catch((error) => {
        console.error("❌ Error submitting recommendation:", error);
        alert("Failed to fetch recommendation. Check backend.");
      });

  };

  return (
    <div className="container">
      <h1>Course Recommendation</h1>
      {questions.map((q, index) => (
        <div key={index} className="question-block">
          <p className="question-text">{q.question}</p>
          {q.options.map((option, i) => (
            <label key={i} className="option-label">
              <input
                type="radio"
                name={`question-${index}`}
                value={option}
                onChange={() => setResponses({ ...responses, [index]: option })}
              />
              {option}
            </label>
          ))}
        </div>
      ))}
      <button onClick={handleSubmit} className="submit-btn">Submit</button>
    </div>
  );
  
};

export default Questionnaire;