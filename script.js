////////////////////////////////////////////////////////////
//////////////////////// Set-up ////////////////////////////
////////////////////////////////////////////////////////////

//Quick fix for resizing some things for mobile-ish viewers
var mobileScreen = ($(window).innerWidth() < 500 ? true : false);

//Scatterplot
var margin = { left: 80, top: 20, right: 20, bottom: 60 },
    width = Math.min($("#chart").width(), 480) - margin.left - margin.right,
    height = width * 2 / 3;

var svg = d3.select("#chart").append("svg")
    .attr("width", (width + margin.left + margin.right))
    .attr("height", (height + margin.top + margin.bottom));

var wrapper = svg.append("g").attr("class", "chordWrapper")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

//////////////////////////////////////////////////////
///////////// Initialize Axes & Scales ///////////////
//////////////////////////////////////////////////////

var opacityCircles = 0.7,
    maxDistanceFromPoint = 50;

//Set the color for each region
var color = d3.scale.ordinal()
    .range(["#EFB605", "#E01A25", "#C20049", "#66489F", "#2074A0", "#10A66E", "#7EB852"])
    // ,"#991C71", "#7EB852"])
    .domain(["Bloom", "OpenLlama", "Llama", "OLMo", "Gemma", "Mistral", "Llama3"]);

//Set the new x axis range
var xScale = d3.scale.linear()
    .range([0, width])
    .domain([-0.01, d3.extent(countries, function (d) { return d.LanguageScore; })[1]])
    .nice();

//.domain(d3.extent(countries, function(d) { return d.LanguageScore; }))
//.nice();
//Set new x-axis
var xAxis = d3.svg.axis()
    .orient("bottom")
    .ticks(5) // Adjust number of ticks as needed for readability
    .tickFormat(d3.format(",.2f")) // Formats the tick as a floating-point number without any decimal places
    .scale(xScale);

// Append the x-axis
wrapper.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(" + 0 + "," + height + ")")
    .call(xAxis)
    .selectAll("g")
    .append("line")
    .classed("grid-line", true) // Add a class for styling if needed
    .attr("stroke", "#edede9")  // Optional: style directly here or through CSS
    .attr("opacity", 0.2)
    .attr("x1", 0)
    .attr("x2", 0)
    .attr("y1", 0)
    .attr("y2", -height);

//Set the new y axis range
var yScale = d3.scale.linear()
    .range([height, 0])
    .domain(d3.extent(countries, function (d) { return d.AlignmentScore; }))
    .nice();
var yAxis = d3.svg.axis()
    .orient("left")
    .ticks(6)  // Set the number of ticks as needed
    .tickFormat(d3.format(".2f"))  // Formats the tick with two decimal places
    .scale(yScale);

// Append the y-axis
wrapper.append("g")
    .attr("class", "y axis")
    .attr("transform", "translate(" + 0 + "," + 0 + ")")
    .call(yAxis)
    .selectAll("g")
    .append("line")
    .classed("grid-line", true) // Add a class for styling if needed
    .attr("stroke", "#e4e4e4")  // Optional: style directly here or through CSS
    .attr("opacity", 0.2)
    .attr("x1", 0)
    .attr("x2", width)
    .attr("y1", 0)
    .attr("y2", 0);

//Scale for the bubble size
var rScale = d3.scale.sqrt()
    .range([mobileScreen ? 1 : 2, mobileScreen ? 10 : 16])
    .domain(d3.extent(countries, function (d) { return d.ParameterSize; }));


//////////////////////////////////////////////////////
///////////////// Initialize Labels //////////////////
//////////////////////////////////////////////////////

//Set up X axis label
wrapper.append("g")
    .attr("class", "x title")
    .attr("transform", "translate(" + (width - 25) + "," + (height + 40) + ")")
    .append("text")
    .attr("text-anchor", "end")
    .style("font-size", (mobileScreen ? 10 : 14) + "px")
    .text("LANGUAGE modeling score")
    .append("tspan")
    .style("font-size", (mobileScreen ? 8 : 12) + "px")
    .style("font-weight", "normal")
    .style("font-family", "Courier")
    .text(" (1 - bits-per-byte)");


//Set up y axis label
wrapper.append("g")
    .attr("class", "y title")
    .attr("transform", "translate(-60, 60) rotate(-90)")
    .append("text")
    .attr("text-anchor", "end")
    .style("font-size", (mobileScreen ? 10 : 14) + "px")
    .text("Alignment to VISION")

wrapper.append("g")
    .attr("class", "y title")
    .attr("transform", "translate(-40, 25) rotate(-90)")
    .append("text")
    .attr("text-anchor", "end")
    .style("font-size", (mobileScreen ? 8 : 12) + "px")
    .append("tspan")
    .style("font-weight", "normal")
    .style("font-family", "Courier")
    .text(" (kernel alignment to DINOv2)");

//////////////////////////////////////////////////////////////
//////////////////// Set-up voronoi //////////////////////////
//////////////////////////////////////////////////////////////

//Initiate the voronoi function
//Use the same variables of the data in the .x and .y as used in the cx and cy of the circle call
//The clip extent will make the boundaries end nicely along the chart area instead of splitting up the entire SVG
//(if you do not do this it would mean that you already see a tooltip when your mouse is still in the axis area, which is confusing)
var voronoi = d3.geom.voronoi()
    .x(function (d) { return xScale(d.LanguageScore); })
    .y(function (d) { return yScale(d.AlignmentScore); })
    .clipExtent([[0, 0], [width, height]]);

var voronoiCells = voronoi(countries);

////////////////////////////////////////////////////////////
///////////// Circles to capture close mouse event /////////
////////////////////////////////////////////////////////////

//Create wrapper for the voronoi clip paths
var clipWrapper = wrapper.append("defs")
    .attr("class", "clipWrapper");

clipWrapper.selectAll(".clip")
    .data(voronoiCells)
    .enter().append("clipPath")
    .attr("class", "clip")
    .attr("id", function (d) { return "clip-" + d.point.ModelCode; })
    .append("path")
    .attr("class", "clip-path-circle")
    .attr("d", function (d) { return "M" + d.join(",") + "Z"; });

//Initiate a group element for the circles
var circleClipGroup = wrapper.append("g")
    .attr("class", "circleClipWrapper");

//Place the larger circles to eventually capture the mouse
var circlesOuter = circleClipGroup.selectAll(".circle-wrapper")
    .data(countries.sort(function (a, b) { return b.ParameterSize > a.ParameterSize; }))
    .enter().append("circle")
    .attr("class", function (d, i) { return "circle-wrapper " + d.ModelCode; })
    .attr("clip-path", function (d) { return "url(#clip-" + d.ModelCode + ")"; })
    .style("clip-path", function (d) { return "url(#clip-" + d.ModelCode + ")"; })
    .attr("cx", function (d) { return xScale(d.LanguageScore); })
    .attr("cy", function (d) { return yScale(d.AlignmentScore); })
    .attr("r", maxDistanceFromPoint)
    .on("mouseover", showTooltip)
    .on("mouseout", removeTooltip);;

////////////////////////////////////////////////////////////
/////////////////// Scatterplot Circles ////////////////////
////////////////////////////////////////////////////////////

//Initiate a group element for the circles
var circleGroup = wrapper.append("g")
    .attr("class", "circleWrapper");

//Place the country circles
circleGroup.selectAll("countries")
    .data(countries.sort(function (a, b) { return b.ParameterSize > a.ParameterSize; })) //Sort so the biggest circles are below
    .enter().append("circle")
    .attr("class", function (d, i) { return "countries " + d.ModelCode; })
    .attr("cx", function (d) { return xScale(d.LanguageScore); })
    .attr("cy", function (d) { return yScale(d.AlignmentScore); })
    .attr("r", function (d) { return rScale(2 * d.ParameterSize); })
    .style("opacity", opacityCircles)
    .style("fill", function (d) { return color(d.ModelFamily); });

///////////////////////////////////////////////////////////////////////////
///////////////////////// Create the Legend////////////////////////////////
///////////////////////////////////////////////////////////////////////////

if (!mobileScreen) {
    //Legend
    var legendMargin = { left: 0, top: 10, right: 5, bottom: 10 },
        legendWidth = 145,
        legendHeight = 270;

    var svgLegend = d3.select("#legend").append("svg")
        .attr("width", (legendWidth + legendMargin.left + legendMargin.right))
        .attr("height", (legendHeight + legendMargin.top + legendMargin.bottom));

    var legendWrapper = svgLegend.append("g").attr("class", "legendWrapper")
        .attr("transform", "translate(" + legendMargin.left + "," + legendMargin.top + ")");

    var rectSize = 15, //dimensions of the colored square
        rowHeight = 20, //height of a row in the legend
        maxWidth = 144; //widht of each row

    //Create container per rect/text pair
    var legend = legendWrapper.selectAll('.legendSquare')
        .data(color.range())
        .enter().append('g')
        .attr('class', 'legendSquare')
        .attr("transform", function (d, i) { return "translate(" + 0 + "," + (i * rowHeight) + ")"; })
        .style("cursor", "pointer")
        .on("mouseover", selectLegend(0.02))
        .on("mouseout", selectLegend(opacityCircles));

    //Non visible white rectangle behind square and text for better hover
    legend.append('rect')
        .attr('width', maxWidth)
        .attr('height', rowHeight)
        .style('fill', "white");
    //Append small squares to Legend
    legend.append('rect')
        .attr('width', rectSize)
        .attr('height', rectSize)
        .style('fill', function (d) { return d; });
    //Append text to Legend
    legend.append('text')
        .attr('transform', 'translate(' + 22 + ',' + (rectSize / 2) + ')')
        .attr("class", "legendText")
        .style("font-size", "10px")
        .attr("dy", ".35em")
        .text(function (d, i) { return color.domain()[i]; });

    //Create g element for bubble size legend
    var bubbleSizeLegend = legendWrapper.append("g")
        .attr("transform", "translate(" + (legendWidth / 2 - 30) + "," + (color.domain().length * rowHeight + 20) + ")");
    //Draw the bubble size legend
    bubbleLegend(bubbleSizeLegend, rScale, legendSizes = [7, 30, 70], legendName = "Parameters");
}//if !mobileScreen
else {
    d3.select("#legend").style("display", "none");
}

//////////////////////////////////////////////////////
/////////////////// Bubble Legend ////////////////////
//////////////////////////////////////////////////////

function bubbleLegend(wrapperVar, scale, sizes, titleName) {

    var legendSize1 = sizes[0],
        legendSize2 = sizes[1],
        legendSize3 = sizes[2],
        legendCenter = 0,
        legendBottom = 60,
        legendLineLength = 25,
        textPadding = 5,
        numFormat = d3.format(",");

    wrapperVar.append("text")
        .attr("class", "legendTitle")
        .attr("transform", "translate(" + legendCenter + "," + 0 + ")")
        .attr("x", 0 + "px")
        .attr("y", 0 + "px")
        .attr("dy", "1em")
        .text(titleName);

    wrapperVar.append("circle")
        .attr('r', scale(2 * legendSize1))
        .attr('class', "legendCircle")
        .attr('cx', legendCenter)
        .attr('cy', (legendBottom - scale(legendSize1)));
    wrapperVar.append("circle")
        .attr('r', scale(2 * legendSize2))
        .attr('class', "legendCircle")
        .attr('cx', legendCenter)
        .attr('cy', (legendBottom - scale(legendSize2)));
    wrapperVar.append("circle")
        .attr('r', scale(2 * legendSize3))
        .attr('class', "legendCircle")
        .attr('cx', legendCenter)
        .attr('cy', (legendBottom - scale(legendSize3)));

    wrapperVar.append("line")
        .attr('class', "legendLine")
        .attr('x1', legendCenter)
        .attr('y1', (legendBottom - 2 * scale(legendSize1)))
        .attr('x2', (legendCenter + legendLineLength))
        .attr('y2', (legendBottom - 2 * scale(legendSize1)));
    wrapperVar.append("line")
        .attr('class', "legendLine")
        .attr('x1', legendCenter)
        .attr('y1', (legendBottom - 2 * scale(legendSize2)))
        .attr('x2', (legendCenter + legendLineLength))
        .attr('y2', (legendBottom - 2 * scale(legendSize2)));
    wrapperVar.append("line")
        .attr('class', "legendLine")
        .attr('x1', legendCenter)
        .attr('y1', (legendBottom - 2 * scale(legendSize3)))
        .attr('x2', (legendCenter + legendLineLength))
        .attr('y2', (legendBottom - 2 * scale(legendSize3)));

    wrapperVar.append("text")
        .attr('class', "legendText")
        .attr('x', (legendCenter + legendLineLength + textPadding))
        .attr('y', (legendBottom - 2 * scale(legendSize1)))
        .attr('dy', '0.25em')
        .text(numFormat(Math.round(legendSize1)) + " B");
    wrapperVar.append("text")
        .attr('class', "legendText")
        .attr('x', (legendCenter + legendLineLength + textPadding))
        .attr('y', (legendBottom - 2 * scale(legendSize2)))
        .attr('dy', '0.25em')
        .text(numFormat(Math.round(legendSize2)) + " B");
    wrapperVar.append("text")
        .attr('class', "legendText")
        .attr('x', (legendCenter + legendLineLength + textPadding))
        .attr('y', (legendBottom - 2 * scale(legendSize3)))
        .attr('dy', '0.25em')
        .text(numFormat(Math.round(legendSize3)) + " B");

}//bubbleLegend

///////////////////////////////////////////////////////////////////////////
//////////////////// Hover function for the legend ////////////////////////
///////////////////////////////////////////////////////////////////////////

//Decrease opacity of non selected circles when hovering in the legend
function selectLegend(opacity) {
    return function (d, i) {
        var chosen = color.domain()[i];

        wrapper.selectAll(".countries")
            .filter(function (d) { return d.ModelFamily != chosen; })
            .transition()
            .style("opacity", opacity);
    };
}//function selectLegend

///////////////////////////////////////////////////////////////////////////
/////////////////// Hover functions of the circles ////////////////////////
///////////////////////////////////////////////////////////////////////////

//Hide the tooltip when the mouse moves away
function removeTooltip(d, i) {

    //Save the chosen circle (so not the voronoi)
    var element = d3.selectAll(".countries." + d.ModelCode);

    //Fade out the bubble again
    element.style("opacity", opacityCircles);

    //Hide tooltip
    $('.popover').each(function () {
        $(this).remove();
    });

    //Fade out guide lines, then remove them
    d3.selectAll(".guide")
        .transition().duration(200)
        .style("opacity", 0)
        .remove();

}//function removeTooltip

//Show the tooltip on the hovered over slice
function showTooltip(d, i) {

    //Save the chosen circle (so not the voronoi)
    var element = d3.selectAll(".countries." + d.ModelCode);

    //Define and show the tooltip
    $(element).popover({
        placement: 'auto top',
        container: '#chart',
        trigger: 'manual',
        html: true,
        content: function () {
            return "<span style='font-size: 11px; text-align: center; font-family: graphik'>" + d.Model + "</span>";
        }
    });
    $(element).popover('show');

    //Make chosen circle more visible
    element.style("opacity", 1);

    //Place and show tooltip
    var x = +element.attr("cx"),
        y = +element.attr("cy"),
        color = element.style("fill");

    //Append lines to bubbles that will be used to show the precise data points

    //vertical line
    wrapper
        .append("line")
        .attr("class", "guide")
        .attr("x1", x)
        .attr("x2", x)
        .attr("y1", y)
        .attr("y2", height - 20)
        .style("stroke", color)
        .style("opacity", 0)
        .transition().duration(200)
        .style("opacity", 0.5);
    //Value on the axis
    wrapper
        .append("text")
        .attr("class", "guide")
        .attr("x", x)
        .attr("y", height - 5)
        .style("fill", color)
        .style("opacity", 0)
        .style("text-anchor", "middle")
        .text(d3.format(".3f")(d.LanguageScore))
        .transition().duration(200)
        .style("opacity", 0.5);

    //horizontal line
    wrapper
        .append("line")
        .attr("class", "guide")
        .attr("x1", x)
        .attr("x2", 42)
        .attr("y1", y)
        .attr("y2", y)
        .style("stroke", color)
        .style("opacity", 0)
        .transition().duration(200)
        .style("opacity", 0.5);
    //Value on the axis
    wrapper
        .append("text")
        .attr("class", "guide")
        .attr("x", 35)
        .attr("y", y)
        .attr("dy", "0.35em")
        .style("fill", color)
        .style("opacity", 0)
        .style("text-anchor", "end")
        .text(d3.format(".3f")(d.AlignmentScore))
        .transition().duration(200)
        .style("opacity", 0.5);

}//function showTooltip
