// let menu = document.querySelector('#menu-bar');
// let nav = document.querySelector('.nav');

// menu.onclick = () =>{
//     menu.classList.toggle('fa-times');
//     nav.classList.toggle('active');
// }

// let section = document.querySelectorAll('section');
// let navLinks = document.querySelectorAll('header .nav a');

// window.onscroll = () =>{

//     menu.classList.remove('fa-times');
//     nav.classList.remove('active');

//     section.forEach(sec =>{

//         let top = window.scrollY;
//         let height = sec.offsetHeight;
//         let offset = sec.offsetTop - 150;
//         let id = sec.getAttribute('id');

//         if(top >= offset && top < offset + height){
//             navLinks.forEach(links =>{
//                 links.classList.remove('active');
//                 document.querySelector('header .nav a[href*='+id+']').classList.add('active');
//             });
//         };
//     });

// }





// document.addEventListener("DOMContentLoaded", function () {
//     const menuBar = document.getElementById("menu-bar");
//     const nav = document.querySelector("header .nav");

//     if (menuBar && nav) {
//         menuBar.addEventListener("click", function () {
//             nav.classList.toggle("show-nav");
//         });

//         // Optional: Close nav when a link is clicked (mobile usability)
//         const navLinks = nav.querySelectorAll("a");
//         navLinks.forEach(link => {
//             link.addEventListener("click", () => {
//                 nav.classList.remove("show-nav");
//             });
//         });
//     }
// });



// document.addEventListener("DOMContentLoaded", function () {
//     const menu = document.querySelector('#menu-bar');
//     const nav = document.querySelector('.nav');

//     if (menu && nav) {
//         menu.addEventListener('click', () => {
//             nav.classList.toggle('active');
//         });
//     }
// });



console.log("main.js is loaded");

document.addEventListener("DOMContentLoaded", function () {
    const menu = document.querySelector('#menu-bar');
    const nav = document.querySelector('.nav');

    if (menu && nav) {
        console.log("menu and nav found");

        menu.addEventListener('click', () => {
            console.log("hamburger clicked");
            nav.classList.toggle('active');
        });
    } else {
        console.warn('menu or nav not found!');
    }
});



