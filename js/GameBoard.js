/*

    Board Scale: 3 --> 300 x 300
    Piece Scale: 9
    Vertical Increment: 600 / (n=6) = 100
    Horizontal Increment: 450, but Space between boards: 150

*/


renderer.setClearColor( 0x050a1f);


GameBoard.prototype = {
    
    initPieces: function(){
        //
        // Logic
        //
        
        // Instantiate pieces
        const range = n => [...Array(n)].map((_, i) => i);
        const rangeIn = dims => {
          if (!dims.length) return new Piece();
          return range(dims[0]).map(_ => rangeIn(dims.slice(1)));
        };
        
        const pieces = rangeIn([this.n, this.n, this.n, this.n])
        
        //
        // Graphics
        //
		return pieces;
    },
    
	// GameBoard.move should only be called from MoveManager. All move data received from server should pass through MoveManager.
    move: function(x0, y0, z0, w0, x1, y1, z1, w1){
		this.graphics.hidePossibleMoves(); // TODO: replace with pointer deselection
		const metaData = {}
		
		const targetPiece = this.pieces[x1][y1][z1][w1];
		if(targetPiece.type){
			Object.assign(metaData, {capturedPiece: targetPiece});
		}
        const piece = this.pieces[x0][y0][z0][w0];
		
		this.graphics.moveMesh(piece, x1, y1, z1, w1);
        
		this.pieces[x0][y0][z0][w0] = new Piece(); // Remove game object from board
		this.removePiece(x1, y1, z1, w1); // Immediately remove target game object and sprite (do nothing if square is empty)
        this.pieces[x1][y1][z1][w1] = piece; // Replace object in target square with moved piece
        Object.assign(metaData, piece.update(this.pieces, x0, y0, z0, w0, x1, y1, z1, w1));
		
		if(piece.type === 'pawn' && this.isOnPromotionSquare(x1, y1, z1, w1)){
			// Normal capture logic and animation is still executed,
			// but here we replace the pawn's game object with a Queen game object
			// Notice that we do not use GameObject.removePiece() method because
			// it will also remove its mesh, which we only want once the animation
			// finishes
			
			// The following 2 lines were causing bugs when spamming redo after a pawn promotion. Error was that the queen's mesh is undefined. The queen's .mesh property was not being set until the animation was complete. The fix was to create the mesh and set the queen's .mesh property immediately in the promotion logic, and only make it appear (add it to piecesContainer) when the animation is finished.
//			queen = new Queen(piece.team);
//			this.pieces[x1][y1][z1][w1] = queen;
			let queen = this.spawnPiece(Queen, piece.team, x1, y1, z1, w1, false);
			piece.descendant = queen;
			
			Object.assign(metaData, {promotion: true, newPiece: queen, oldPiece: piece});
		}
		
        return metaData;
    },
	
	undo: function(move){
		this.graphics.hidePossibleMoves(); // TODO: replace with pointer deselection
		const pieceInOriginalLoc = this.pieces[move.x0][move.y0][move.z0][move.w0];
		if(pieceInOriginalLoc.type){
			console.error('Unknown error. A piece is already located in original location')
		}
		const originalPiece = move.metaData.promotion ? move.metaData.oldPiece : this.pieces[move.x1][move.y1][move.z1][move.w1];
		this.pieces[move.x0][move.y0][move.z0][move.w0] = originalPiece;
		
		const capturedPiece = move.metaData.capturedPiece || new Piece();
		if(move.metaData.promotion){
			this.graphics.respawnMesh(originalPiece, move.x1, move.y1, move.z1, move.w1);
			this.removePiece(move.x1, move.y1, move.z1, move.w1); // TODO: THIS IS SCARY. If bugs occur separate the graphics component of removePiece into a new method. The current implementaiton might cause errors...
		}
		this.pieces[move.x1][move.y1][move.z1][move.w1] = capturedPiece;
		
		if (move.metaData.capturedPiece) {
			this.graphics.respawnMesh(capturedPiece, move.x1, move.y1, move.z1, move.w1); 
		}
		if (move.metaData.justMoved) {
			originalPiece.hasMoved = false;
		}
		this.graphics.moveMesh(originalPiece, move.x0, move.y0, move.z0, move.w0);
	},
	
	applyToAll: function(f){
		for(let x = 0; x < this.pieces.length; x++){
			for(let y = 0; y < this.pieces[0].length; y++){
				for(let z = 0; z < this.pieces[0][0].length; z++){
					for(let w = 0; w < this.pieces[0][0][0].length; w++){
						const piece = this.pieces[x][y][z][w]
						f(piece)
					}
				}
			}
		}
	},
	
	applyToPieces: function(f){
		function onlyPieces(piece){
			if(piece.type)
				f(piece)
		}
		this.applyToAll(onlyPieces)
	},
	
	applyToTeam: function(f, team){
		function onlyTeam(piece){
			if(piece.team === team){
				f(piece)
			}
		}
		this.applyToPieces(onlyTeam)
	},
    
	
	
    isOnPromotionSquare: function(x, y, z, w){
		const piece = this.pieces[x][y][z][w]
		const promotionLoc = piece.team > 0 ? 0 : this.n - 1
		return z === promotionLoc && w === promotionLoc
    },
    
    spawnPiece: function(PieceConstructor, team, x, y, z, w, addToContainer=true){
        const piece = new PieceConstructor(team)
        this.pieces[x][y][z][w] = piece
		this.graphics.setMesh(piece, x, y, z, w, addToContainer);
		return piece;
    },
	
	removePiece: function(x, y, z, w){
		const piece = this.pieces[x][y][z][w]
		this.graphics.removeMesh(piece);
		this.pieces[x][y][z][w] = new Piece();
	},
    
    inBounds: function(x, y, z, w) {
        return x >= 0 && x < this.n && y >= 0 && y < this.n && z >=0 && z < this.n && w >=0 && w < this.n;
    },
	
	package: function() {
		const x = this.pieces;
		const pieces = x.map(y => 
			y.map(z => 
				z.map(w => 
					w.map(piece => piece.package())
				)
			)
		);
		return pieces;
	},
	
	loadPieces: function(piecesArr) {
		this.graphics.abandon(); // remove all meshes from scene
		const Graphics = (this.graphics instanceof BoardGraphics) ? BoardGraphics : EmptyBoardGraphics;
		this.graphics = new Graphics(this) // create new graphics
		
		this.pieces = this.initPieces();
		for (let x = 0; x < this.n; x++) {
			for (let y = 0; y < this.n; y++) {
				for(let z = 0; z < this.n; z++) {
					for(let w = 0; w < this.n; w++) {
						const read = piecesArr[x][y][z][w];
						if (read.type) {
							const PieceConstructor = Piece.typeToConstructor[read.type];
							let piece = this.spawnPiece(PieceConstructor, read.team, x, y, z, w);
							Object.assign(piece, read);
						}
						
					}
				}
			}
		}
	},
    
    test: function(x, y, z, w){
        
        if(x == null) x = getRandomInteger(0, this.n)
        if(y == null) y = getRandomInteger(0, this.n)
        if(z == null) z = getRandomInteger(0, this.n)
        if(w == null) w = getRandomInteger(0, this.n)
		
        this.spawnPiece(Rook, 0, 0, 0, 0, 0)
        this.spawnPiece(Knight, 0, 1, 0, 0, 0)
        this.spawnPiece(Knight, 0, 2, 0, 0, 0)
        this.spawnPiece(Rook, 0, 3, 0, 0, 0)
        this.spawnPiece(Bishop, 0, 0, 1, 0, 0)
        this.spawnPiece(Pawn, 0, 1, 1, 0, 0)
        this.spawnPiece(Pawn, 0, 2, 1, 0, 0)
        this.spawnPiece(Bishop, 0, 3, 1, 0, 0)
        this.spawnPiece(Bishop, 0, 0, 2, 0, 0)
        this.spawnPiece(Queen, 0, 1, 2, 0, 0)
        this.spawnPiece(King, 0, 2, 2, 0, 0)
        this.spawnPiece(Bishop, 0, 3, 2, 0, 0)
        this.spawnPiece(Rook, 0, 0, 3, 0, 0)
        this.spawnPiece(Knight, 0, 1, 3, 0, 0)
        this.spawnPiece(Knight, 0, 2, 3, 0, 0)
        this.spawnPiece(Rook, 0, 3, 3, 0, 0)
        
        this.spawnPiece(Pawn, 0, 0, 0, 1, 0)
        this.spawnPiece(Pawn, 0, 1, 0, 1, 0)
        this.spawnPiece(Pawn, 0, 2, 0, 1, 0)
        this.spawnPiece(Pawn, 0, 3, 0, 1, 0)
        this.spawnPiece(Pawn, 0, 0, 1, 1, 0)
        this.spawnPiece(Pawn, 0, 1, 1, 1, 0)
        this.spawnPiece(Pawn, 0, 2, 1, 1, 0)
        this.spawnPiece(Pawn, 0, 3, 1, 1, 0)
        this.spawnPiece(Pawn, 0, 0, 2, 1, 0)
        this.spawnPiece(Pawn, 0, 1, 2, 1, 0)
        this.spawnPiece(Pawn, 0, 2, 2, 1, 0)
        this.spawnPiece(Pawn, 0, 3, 2, 1, 0)
        this.spawnPiece(Pawn, 0, 0, 3, 1, 0)
        this.spawnPiece(Pawn, 0, 1, 3, 1, 0)
        this.spawnPiece(Pawn, 0, 2, 3, 1, 0)
        this.spawnPiece(Pawn, 0, 3, 3, 1, 0)
        
        this.spawnPiece(Pawn, 0, 0, 0, 0, 1)
        this.spawnPiece(Pawn, 0, 1, 0, 0, 1)
        this.spawnPiece(Pawn, 0, 2, 0, 0, 1)
        this.spawnPiece(Pawn, 0, 3, 0, 0, 1)
        this.spawnPiece(Pawn, 0, 0, 1, 0, 1)
        this.spawnPiece(Pawn, 0, 1, 1, 0, 1)
        this.spawnPiece(Pawn, 0, 2, 1, 0, 1)
        this.spawnPiece(Pawn, 0, 3, 1, 0, 1)
        this.spawnPiece(Pawn, 0, 0, 2, 0, 1)
        this.spawnPiece(Pawn, 0, 1, 2, 0, 1)
        this.spawnPiece(Pawn, 0, 2, 2, 0, 1)
        this.spawnPiece(Pawn, 0, 3, 2, 0, 1)
        this.spawnPiece(Pawn, 0, 0, 3, 0, 1)
        this.spawnPiece(Pawn, 0, 1, 3, 0, 1)
        this.spawnPiece(Pawn, 0, 2, 3, 0, 1)
        this.spawnPiece(Pawn, 0, 3, 3, 0, 1)
        
        this.spawnPiece(Pawn, 0, 0, 0, 1, 1)
        this.spawnPiece(Pawn, 0, 1, 0, 1, 1)
        this.spawnPiece(Pawn, 0, 2, 0, 1, 1)
        this.spawnPiece(Pawn, 0, 3, 0, 1, 1)
        this.spawnPiece(Pawn, 0, 0, 1, 1, 1)
        this.spawnPiece(Pawn, 0, 1, 1, 1, 1)
        this.spawnPiece(Pawn, 0, 2, 1, 1, 1)
        this.spawnPiece(Pawn, 0, 3, 1, 1, 1)
        this.spawnPiece(Pawn, 0, 0, 2, 1, 1)
        this.spawnPiece(Pawn, 0, 1, 2, 1, 1)
        this.spawnPiece(Pawn, 0, 2, 2, 1, 1)
        this.spawnPiece(Pawn, 0, 3, 2, 1, 1)
        this.spawnPiece(Pawn, 0, 0, 3, 1, 1)
        this.spawnPiece(Pawn, 0, 1, 3, 1, 1)
        this.spawnPiece(Pawn, 0, 2, 3, 1, 1)
        this.spawnPiece(Pawn, 0, 3, 3, 1, 1)
        
        const l = this.n - 1 // Back row (black)
        const m = l - 1 // Front Row (black)
        
        this.spawnPiece(Rook, 1, 0, 0, l, l)
        this.spawnPiece(Knight, 1, 1, 0, l, l)
        this.spawnPiece(Knight, 1, 2, 0, l, l)
        this.spawnPiece(Rook, 1, 3, 0, l, l)
        this.spawnPiece(Bishop, 1, 0, 1, l, l)
        this.spawnPiece(Pawn, 1, 1, 1, l, l)
        this.spawnPiece(Pawn, 1, 2, 1, l, l)
        this.spawnPiece(Bishop, 1, 3, 1, l, l)
        this.spawnPiece(Bishop, 1, 0, 2, l, l)
        this.spawnPiece(Queen, 1, 1, 2, l, l)
        this.spawnPiece(King, 1, 2, 2, l, l)
        this.spawnPiece(Bishop, 1, 3, 2, l, l)
        this.spawnPiece(Rook, 1, 0, 3, l, l)
        this.spawnPiece(Knight, 1, 1, 3, l, l)
        this.spawnPiece(Knight, 1, 2, 3, l, l)
        this.spawnPiece(Rook, 1, 3, 3, l, l)
        
        this.spawnPiece(Pawn, 1, 0, 0, m, l)
        this.spawnPiece(Pawn, 1, 1, 0, m, l)
        this.spawnPiece(Pawn, 1, 2, 0, m, l)
        this.spawnPiece(Pawn, 1, 3, 0, m, l)
        this.spawnPiece(Pawn, 1, 0, 1, m, l)
        this.spawnPiece(Pawn, 1, 1, 1, m, l)
        this.spawnPiece(Pawn, 1, 2, 1, m, l)
        this.spawnPiece(Pawn, 1, 3, 1, m, l)
        this.spawnPiece(Pawn, 1, 0, 2, m, l)
        this.spawnPiece(Pawn, 1, 1, 2, m, l)
        this.spawnPiece(Pawn, 1, 2, 2, m, l)
        this.spawnPiece(Pawn, 1, 3, 2, m, l)
        this.spawnPiece(Pawn, 1, 0, 3, m, l)
        this.spawnPiece(Pawn, 1, 1, 3, m, l)
        this.spawnPiece(Pawn, 1, 2, 3, m, l)
        this.spawnPiece(Pawn, 1, 3, 3, m, l)
        
        this.spawnPiece(Pawn, 1, 0, 0, l, m)
        this.spawnPiece(Pawn, 1, 1, 0, l, m)
        this.spawnPiece(Pawn, 1, 2, 0, l, m)
        this.spawnPiece(Pawn, 1, 3, 0, l, m)
        this.spawnPiece(Pawn, 1, 0, 1, l, m)
        this.spawnPiece(Pawn, 1, 1, 1, l, m)
        this.spawnPiece(Pawn, 1, 2, 1, l, m)
        this.spawnPiece(Pawn, 1, 3, 1, l, m)
        this.spawnPiece(Pawn, 1, 0, 2, l, m)
        this.spawnPiece(Pawn, 1, 1, 2, l, m)
        this.spawnPiece(Pawn, 1, 2, 2, l, m)
        this.spawnPiece(Pawn, 1, 3, 2, l, m)
        this.spawnPiece(Pawn, 1, 0, 3, l, m)
        this.spawnPiece(Pawn, 1, 1, 3, l, m)
        this.spawnPiece(Pawn, 1, 2, 3, l, m)
        this.spawnPiece(Pawn, 1, 3, 3, l, m)
        
        this.spawnPiece(Pawn, 1, 0, 0, m, m)
        this.spawnPiece(Pawn, 1, 1, 0, m, m)
        this.spawnPiece(Pawn, 1, 2, 0, m, m)
        this.spawnPiece(Pawn, 1, 3, 0, m, m)
        this.spawnPiece(Pawn, 1, 0, 1, m, m)
        this.spawnPiece(Pawn, 1, 1, 1, m, m)
        this.spawnPiece(Pawn, 1, 2, 1, m, m)
        this.spawnPiece(Pawn, 1, 3, 1, m, m)
        this.spawnPiece(Pawn, 1, 0, 2, m, m)
        this.spawnPiece(Pawn, 1, 1, 2, m, m)
        this.spawnPiece(Pawn, 1, 2, 2, m, m)
        this.spawnPiece(Pawn, 1, 3, 2, m, m)
        this.spawnPiece(Pawn, 1, 0, 3, m, m)
        this.spawnPiece(Pawn, 1, 1, 3, m, m)
        this.spawnPiece(Pawn, 1, 2, 3, m, m)
        this.spawnPiece(Pawn, 1, 3, 3, m, m)
        
    }
    
}


EmptyBoardGraphics.prototype = {
	
	getBoundingBox: function() {},
	
	getCenter: function(){},
	
	boardCoordinates: function(x, y, z, w){},
	
	worldCoordinates: function(pos){},
	
	showPossibleMoves: function(locations, piece, materialScheme={}, canRayCast){},
    
    showPossibleMoves2: function(x, y, z, w, materialScheme){},
    
    hidePossibleMoves: function(objectName='possibleMoves'){},
	
	moveMesh: function(piece, x1, y1, z1, w1){},
	
	createMesh: function(typeString, team, x, y, z, w, addToContainer=true){},
	
	setMesh: function(piece, x, y, z, w, addToContainer=true){},
	
	removeMesh: function(piece){},
	
	respawnMesh: function(piece){},
	
	setSelectability: function(piece, canMove) {},
	
	addToPiecesContainer: function(mesh) {},
	
	abandon: function() {}
	
}





BoardGraphics.prototype = {
	
	getBoundingBox: function() {
		const offset = this.boardSize / 2;
		const originOffset = new THREE.Vector3(offset, 0, -offset);
		const logicalLocalOrigin = originOffset.clone().multiplyScalar(-1);
		const globalLLO = logicalLocalOrigin.clone().add(this.mesh.position);
		
		const globalBottomLeft = logicalLocalOrigin.clone().add(this.mesh.position);
		const halfDiagonal = this.getCenter().sub(globalLLO);
		const extraHeight = new THREE.Vector3(0, this.verticalIncrement, 0);
		const globalTopRight = this.getCenter().clone().add(halfDiagonal).clone().add(extraHeight);
		
		const extraSpace = new THREE.Vector3(this.squareSize * 0, 0, -this.squareSize)
		
		return {
			bottomLeft: globalBottomLeft.sub(extraSpace),
			topRight: globalTopRight.add(extraSpace)
		};
	},
	
	getCenter: function(){
		const numHalfBoards = (this.n - 1) / 2;
		const localZ = -(this.boardSize + this.horizontalGap) * numHalfBoards;
		const localY = this.verticalIncrement * numHalfBoards;
		const localCenter = new THREE.Vector3(0, localY, localZ);
		return localCenter.add(this.mesh.position);
	},
	
	boardCoordinates: function(x, y, z, w){
        
        // Get world coordinates from board coordinates
        
        const zero = new THREE.Vector3((0.5 * this.squareSize) - (0.5 * this.squareSize * this.n), 0, (0.5 * this.squareSize * this.n) - (0.5 * this.squareSize))
        
        const xShift = x * this.squareSize
        const yShift = y * this.verticalIncrement + this.EPSILON
        const zShift = -(z * this.squareSize + w * this.horizontalIncrement)
        const translation = new THREE.Vector3(xShift, yShift, zShift)
        
        return zero.add(translation).add(this.boardContainer.position)
        
    },
	
	worldCoordinates: function(pos){
        
        // Get board coordinates from world coordinates
        const zero = new THREE.Vector3((0.5 * this.squareSize) - (0.5 * this.squareSize * this.n), 0, (0.5 * this.squareSize * this.n) - (0.5 * this.squareSize))
        pos = pos.clone().sub(zero).sub(this.boardContainer.position)
        
        let x = Math.floor(pos.x / this.squareSize)
        let y = Math.floor(pos.y / this.verticalIncrement)
        let numGaps = Math.floor(-pos.z / this.horizontalIncrement)
        let z = Math.floor((-pos.z - (numGaps * this.horizontalIncrement)) / this.squareSize)
        let w = numGaps
        
//        pos.sub(this.boardContainer.position)
        
        return new THREE.Vector4(x, y, z, w)
    },
	
	showPossibleMoves: function(locations, piece, materialScheme={}, canRayCast){
		materialScheme = Object.assign({
			0: {
				moveMaterial: Models.materials.green,
				attackMaterial: Models.materials.red
			},
			1: {
				moveMaterial: Models.materials.green,
				attackMaterial: Models.materials.red
			},
		}, materialScheme)
		
        this.hidePossibleMoves();
		
        locations.forEach(pos => {
			
			coordinates = this.boardCoordinates(pos.x, pos.y, pos.z, pos.w)
			let material;
			let shadowPiece;
			if(pos.possibleCapture){
				const attackedPiece = this.gameBoard.pieces[pos.x][pos.y][pos.z][pos.w]
				material = materialScheme[piece.team].attackMaterial;
				shadowPiece = Models.createMesh(attackedPiece.type, material, coordinates.x, coordinates.y, coordinates.z, 1, canRayCast)
				if(attackedPiece.team === 0){
					rotateObject(shadowPiece, 0, 180, 0)
				}
			} else {
				let material = materialScheme[piece.team].moveMaterial;
				shadowPiece = Models.createMesh(piece.type, material, coordinates.x, coordinates.y, coordinates.z, 1, canRayCast)
			}
				
			this.possibleMovesContainer.add(shadowPiece)
            
            			
        });
        
    },
    
    showPossibleMoves2: function(x, y, z, w, materialScheme){
      
        const locations = this.pieces[x][y][z][w].getPossibleMoves(this.pieces, x, y, z, w)
        
        this.showPossibleMoves(locations, piece, materialScheme, true)
        
    },
    
    hidePossibleMoves: function(objectName='possibleMoves'){
        while(this.possibleMovesContainer.children.length){
			const mesh = this.possibleMovesContainer.children[0]
			Selector.unhighlight(mesh)
            this.possibleMovesContainer.remove(mesh);
        }
    },
	
	moveMesh: function(piece, x1, y1, z1, w1){
		const currentMeshCoords = piece.mesh.position
        const newMeshCoords = this.boardCoordinates(x1, y1, z1, w1)
        
        const frames = 12
        const interpolatedCoords = Animation.linearInterpolate(currentMeshCoords, newMeshCoords, frames)
        Animation.addToQueue(animationQueue, piece.mesh, interpolatedCoords)
        piece.mesh.canRayCast = false // temporarily disable piece's ability to be found in rayCast
        
        animationQueue[animationQueue.length - 1].onAnimate = function(){
            piece.mesh.canRayCast = true // re-enable piece's ability to be found in rayCast
            if(piece.type === 'pawn' && this.isOnPromotionSquare(x1, y1, z1, w1)){
				// Normal capture logic and animation is still executed,
				// but here we remove the pawn's sprite and spawn in a queen
				// Notice that we do not use GameObject.removePiece() method because
				// it will also remove its game object, which is logic that should 
				// be separate from graphics
				this.graphics.removeMesh(piece)
				this.graphics.addToPiecesContainer(piece.descendant.mesh); // Only add mesh to scene when animation is finished
//				this.graphics.piecesContainer.add(queen.mesh); 
            }
        }.bind(this.gameBoard)
	},
	
	createMesh: function(typeString, team, x, y, z, w, addToContainer=true){ 
		
		// Create mesh (without game object), add it to the scene, and return the mesh
		const worldPos = this.boardCoordinates(x, y, z, w)
		const material = team === 0 ? Models.materials.white : Models.materials.black
		const mesh = Models.createMesh(typeString, material, worldPos.x, worldPos.y, worldPos.z)
		if(team === 0) rotateObject(mesh, 0, 180, 0)
		if(addToContainer){
			this.piecesContainer.add(mesh)
		}
		
		return mesh
		
	},
	
	setMesh: function(piece, x, y, z, w, addToContainer=true){
//		const worldPos = this.boardCoordinates(x, y, z, w)
//		const mesh = this.createMesh(piece.type, piece.team, worldPos.x, worldPos.y, worldPos.z, worldPos.w);
		const mesh = this.createMesh(piece.type, piece.team, x, y, z, w, addToContainer);
        piece.setMesh(mesh);
	},
	
	removeMesh: function(piece){
		this.piecesContainer.remove(piece.mesh);
	},
	
	respawnMesh: function(piece, x, y, z, w){
		if (piece.mesh) {
			this.piecesContainer.add(piece.mesh);
		} else {
			piece.mesh = this.createMesh(piece.type, piece.team, x, y, z, w, true);
		}
		
	},
	
	setSelectability: function(piece, canMove) {
		piece.mesh.selectable = canMove;
	},
	
	addToPiecesContainer: function(mesh) {
		this.piecesContainer.add(mesh); 
	},
	
	abandon: function() {
		scene.remove(this.mesh)
	}
	
}


function GameBoard(n=4, Graphics=BoardGraphics){
	
    this.n = n;
    
	this.graphics = new Graphics(this);
    this.pieces = this.initPieces();
    
    let halfN = Math.floor((this.n - 1) / 2)
    this.test(halfN, halfN, halfN+2, halfN)
	
	this.setSelectability = function(team, canMove){
		// Enable/Disable piece rayCasting (block user interaction)
		this.applyToTeam(function(piece){
			this.graphics.setSelectability(piece, canMove);
//			piece.mesh.selectable = canMove
		}.bind(this), team)
	};
}

function BoardGraphics(gameBoard) {
//	EmptyBoardGraphics.call(this, gameBoard);
	this.gameBoard = gameBoard;
	this.n = gameBoard.n;
	this.squareSize = 50
	this.boardSize = this.squareSize * this.n;
    this.verticalIncrement = 100 * 1.75
    this.horizontalGap = this.squareSize * 1.6
    this.horizontalIncrement = this.n * this.squareSize + this.horizontalGap
    this.globalLength = this.horizontalIncrement * (this.n - 1)
    this.globalHeight = this.verticalIncrement * this.n
	this.boardHeight = 5;
    this.EPSILON = 1
	
	this.mesh = new THREE.Object3D();
	this.boardContainer = new THREE.Object3D();
    this.piecesContainer = new THREE.Object3D();
    this.possibleMovesContainer = new THREE.Object3D();
	this.boardContainer.name = 'boardContainer';
	this.piecesContainer.name = 'piecesContaier';
	this.possibleMovesContainer.name = 'possibleMovesContainer';
	this.mesh.add(this.boardContainer)
	this.mesh.add(this.piecesContainer)
	this.mesh.add(this.possibleMovesContainer)
	scene.add(this.mesh);
	
	let bottom = 0;
	let left = 0;
	for (let w = 0; w < this.n; w++){
		for(let i = 0; i < this.n; i++){
			let checker = BoardGraphics.checkerboard3d(this.n, this.n * this.squareSize, z=i, w, opacity=0.4, this.boardHeight) // Construct 2D checkerboard planes
			checker.position.set(0, bottom + i*this.verticalIncrement, left - w*this.horizontalIncrement)
			rotateObject(checker, -90, 0, 0)
			this.boardContainer.add(checker)
		}
	}
	
}

BoardGraphics.checkerboard = function(segments=8, boardSize=100, z=0, w=0, opacity=0.5){
    
    let geometry = new THREE.PlaneGeometry(boardSize, boardSize, segments, segments)
//    let geometry = new THREE.BoxGeometry(boardSize, boardSize, 5, segments, segments, 1)
    let materialEven = new THREE.MeshBasicMaterial({color: 0xccccfc})
    let materialOdd = new THREE.MeshBasicMaterial({color: 0x444464})
    let materials = [materialEven, materialOdd]

    materials.forEach(m => {
        m.transparent = true
        m.opacity = opacity
        m.side = THREE.DoubleSide;
    })

    let i = j = 0;

    for(let x = 0; x < segments; x++){
      for(let y = 0; y < segments; y++){
          i = x * segments + y
          j = i * 2
          geometry.faces[j].materialIndex = geometry.faces[j + 1].materialIndex = (x + y + z + w) % 2
      }
    }
	
    return new THREE.Mesh(geometry, new THREE.MultiMaterial(materials))
//    return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color: 0x000000}))
}

BoardGraphics.checkerboard3d = function(segments=8, boardSize=100, z=0, w=0, opacity=0.5, boardHeight=5){
	
	const BOARD_HEIGHT = boardHeight;
	
	let topGeometry = new THREE.PlaneGeometry(boardSize, boardSize, segments, segments);
	let bottomGeometry = new THREE.PlaneGeometry(boardSize, boardSize, segments, segments);
	let materialsTop = [new THREE.MeshBasicMaterial({color: 0xccccfc}), new THREE.MeshBasicMaterial({color: 0x444464})];
	let materialsBottom = [new THREE.MeshBasicMaterial({color: 0xccccfc}), new THREE.MeshBasicMaterial({color: 0x444464})];
	materialsTop.forEach(m => {
        m.transparent = true;
        m.opacity = opacity;
        m.side = THREE.FrontSide;
    });
	materialsBottom.forEach(m => {
        m.transparent = true
        m.opacity = opacity
        m.side = THREE.BackSide;
    });
	for(let x = 0; x < segments; x++){
      for(let y = 0; y < segments; y++){
          i = x * segments + y;
          j = i * 2;
          topGeometry.faces[j].materialIndex = topGeometry.faces[j + 1].materialIndex = (x + y + z + w) % 2;
		  bottomGeometry.faces[j].materialIndex = bottomGeometry.faces[j + 1].materialIndex = (x + y + z + w) % 2
      }
    }
	
	let topMesh = new THREE.Mesh(topGeometry, new THREE.MultiMaterial(materialsTop));
	let bottomMesh = new THREE.Mesh(bottomGeometry, new THREE.MultiMaterial(materialsBottom));
	bottomMesh.position.set(0, 0, -BOARD_HEIGHT);
	
	let sideGeometry = new THREE.PlaneGeometry(BOARD_HEIGHT, boardSize, 1, segments);
	let materialsSide = [new THREE.MeshBasicMaterial({color: 0x444464}), new THREE.MeshBasicMaterial({color: 0xccccfc})];
	materialsSide.forEach(m => {
        m.transparent = true
        m.opacity = opacity
        m.side = THREE.FrontSide;
    });
	for(let x = 0; x < segments; x++){
      for(let y = 0; y < 1; y++){
          i = x * 1 + y;
          j = i * 2;
          sideGeometry.faces[j].materialIndex = sideGeometry.faces[j + 1].materialIndex = ((x + y + z + w) % 2) ^ (segments % 2);
      }
    }
	let sideMesh1 = new THREE.Mesh(sideGeometry, new THREE.MultiMaterial(materialsSide));
	let sideMesh2 = new THREE.Mesh(sideGeometry, new THREE.MultiMaterial(materialsSide));
	let sideMesh3 = new THREE.Mesh(sideGeometry, new THREE.MultiMaterial(materialsSide));
	let sideMesh4 = new THREE.Mesh(sideGeometry, new THREE.MultiMaterial(materialsSide));
	rotateObject(sideMesh1, 0, 90, 0) //front
	rotateObject(sideMesh2, 90, 0, 90) //left
	rotateObject(sideMesh3, 180, -90, 0) //back
	rotateObject(sideMesh4, -90, 0, -90) //right
	sideMesh1.position.set(boardSize/2, 0, -BOARD_HEIGHT/2)
	sideMesh2.position.set(0, -boardSize/2, -BOARD_HEIGHT/2)
	sideMesh3.position.set(-boardSize/2, 0, -BOARD_HEIGHT/2)
	sideMesh4.position.set(0, boardSize/2, -BOARD_HEIGHT/2)
	
	let boxContainer = new THREE.Group();
	boxContainer.add(topMesh);
	boxContainer.add(bottomMesh);
	boxContainer.add(sideMesh1);
	boxContainer.add(sideMesh2);
	boxContainer.add(sideMesh3);
	boxContainer.add(sideMesh4);
	return boxContainer;
	
	
}

function EmptyBoardGraphics(gameBoard) {

}

