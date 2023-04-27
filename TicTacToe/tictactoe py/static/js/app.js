
let n = window.location.href.split('/')[4];

function load_grid(){
    let grid = document.querySelectorAll('.grid')[0];
    let tile = document.querySelectorAll('.tile')[0];

    grid.classList.add("grid-template-rows")
    grid.classList.add("grid-template-columns")
    grid.style["grid-template-columns"]= "1fr ".repeat(n).trim();
    grid.style["grid-template-rows"]= "1fr ".repeat(n).trim();

}
load_grid()
document.documentElement.style.setProperty('--n', n);


class Board {


    constructor() {

        this.tiles = document.querySelectorAll(".tile");

        this.x_img = "<img src='../static/assets/x.png'></img>";
        this.o_img = "<img src='../static/assets/o.png'></img>";

        this.current_turn = 0; // 0 is x

    }

    handleClick(el, i) {
        
        // handles turn and click

        if (this.current_turn == 0) {
            // X turn
            el.setAttribute("ch", "X");
            el.style.padding = "2vh"
            el.innerHTML = this.x_img;
            this.current_turn = 1;
        } else {
            // o turn
            el.setAttribute("ch", "O");
            el.style.padding = "2vh"
            el.innerHTML = this.o_img;
            this.current_turn = 0;
        }

    }

    winCheck() {

        let ls = document.querySelectorAll(".tile");
        const arr = Array.from(ls);
        const data = arr.map(item => item.getAttribute("ch"));

        // console.log(JSON.stringify(data))
        var resp =10;

        const url = "/submit";
        fetch(url, {
                    method: "POST",
                    headers: {
                                    "Content-Type": "application/json",
                                },
                    body: JSON.stringify(data),
                    })
                        .then((response) => response.json())
                        .then((response) => {
                            window.resp = response;
                            console.log(window.resp);

                        });

        return window.resp

        // fetch('https://example.com/data')
        // .then(response => response.json())
        // .then(data => console.log(data));
        // checks if any player has won yet every turn

        }

}

gameBoard = new Board


function tile_clicked(el, i) {


    // console.log("click", el, i);

    gameBoard.handleClick(el, i);

    let won = gameBoard.winCheck();
    console.log(won)
    if (won[0]==true) {

        setTimeout(function() { alert(won[1]) }, 1000)
        console.log("won: ",won[1]);

    }

    // return true
}



// main_game()